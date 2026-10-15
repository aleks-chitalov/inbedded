use opencv::{
    core::{self, Point, Point2f, Scalar, Size, Vector},
    imgcodecs,
    imgproc,
    prelude::*,
    highgui,
    Result
};
use std::f64::consts::PI;

fn test_load() -> Result<()> {
    let image_path = "/home/egorobi/Courses/VLCS/inbedded/crates/decoding/screen.jpeg";
    let src = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if src.empty() {
        println!("Error: Could not open or find the image!");
        return Ok(());
    }
    let mut gray = core::Mat::default();
    imgproc::cvt_color(&src, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut blurred = core::Mat::default();
        imgproc::bilateral_filter(
        &gray,
        &mut blurred,
        21,
        75.0,
        75.0,
        core::BORDER_DEFAULT,
    )?;

    let mut edges = core::Mat::default();
    imgproc::canny(&blurred, &mut edges, 30.0, 100.0, 3, false)?;

    let mut contours = Vector::<Vector<Point>>::new();
    let mut hierarchy = Mat::default();
    
    imgproc::find_contours_with_hierarchy(
        &edges,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut debug_img = src.clone();
    imgproc::draw_contours(
        &mut debug_img,
        &contours,
        -1,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        2, // Thickness
        imgproc::LINE_8,
        &core::no_array(),
        i32::MAX,
        Point::new(0, 0),
    )?;

    highgui::imshow("test", &debug_img)?;
    highgui::wait_key(0)?;
    highgui::destroy_all_windows()?;

    let mut screen_cnt: Option<Vector<Point>> = None;
    let mut max_area = 0.0;

    for i in 0..contours.len() {
        let cnt = contours.get(i)?;
        let area = imgproc::contour_area(&cnt, false)?;

        if area > 1000.0 {
            let arc_len = imgproc::arc_length(&cnt, true)?;
            
            let mut approx = Vector::<Point>::new();
            imgproc::approx_poly_dp(&cnt, &mut approx, 0.02 * arc_len, true)?;

            // Check if it has 4 vertices and is convex
            if approx.len() == 4 && imgproc::is_contour_convex(&approx)? {
                if area > max_area {
                    max_area = area;
                    screen_cnt = Some(approx);
                }
            }
        }
    }

    let screen_points = match screen_cnt {
        Some(pts) => pts,
        None => {
            println!("No screen detected!");
            return Ok(());
        }
    };

    let src_pts = sort_corners(&screen_points)?;

    let dst_width = 800.0;
    let dst_height = 600.0;

    let mut dst_pts = Vector::<Point2f>::new();
    dst_pts.push(Point2f::new(0.0, 0.0));
    dst_pts.push(Point2f::new(dst_width - 1.0, 0.0));
    dst_pts.push(Point2f::new(dst_width - 1.0, dst_height - 1.0));
    dst_pts.push(Point2f::new(0.0, dst_height - 1.0));

    let transform_matrix = imgproc::get_perspective_transform(&src_pts, &dst_pts, core::DECOMP_LU)?;
    let mut warped = Mat::default();
    imgproc::warp_perspective(
        &src,
        &mut warped,
        &transform_matrix,
        Size {
            width: dst_width as i32,
            height: dst_height as i32,
        },
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    highgui::imshow("test", &warped)?;

    highgui::wait_key(0)?;
    highgui::destroy_all_windows()?;

    Ok(())
}

/// Helper function to sort 4 corner points into: Top-Left, Top-Right, Bottom-Right, Bottom-Left
fn sort_corners(pts: &Vector<Point>) -> opencv::Result<Vector<Point2f>> {
    let mut points = vec![];
    for i in 0..pts.len() {
        let p = pts.get(i)?;
        points.push(Point2f::new(p.x as f32, p.y as f32));
    }

    points.sort_by(|a, b| (a.x + a.y).partial_cmp(&(b.x + b.y)).unwrap());
    
    let tl = points[0];
    let br = points[3];

    let mut tr = points[1];
    let mut bl = points[2];

    if (points[1].y - points[1].x) > (points[2].y - points[2].x) {
        bl = points[1];
        tr = points[2];
    }

    let mut sorted = Vector::<Point2f>::new();
    sorted.push(tl);
    sorted.push(tr);
    sorted.push(br);
    sorted.push(bl);

    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        test_load();
    }
}
