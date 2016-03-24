package utils;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import javax.swing.JComponent;

import structure.DEdge;
import structure.DPoint;


public class Lines extends JComponent{
	private HashSet<DEdge> full_edges=null;
	private HashSet<DEdge> mst_edges=null;
	private ArrayList<DPoint> p=null;
	private double xs;
	private double ys;
	public Lines(HashSet<DEdge> full,HashSet<DEdge> mst){
		super();
		full_edges=full;
		mst_edges=mst;
		xs=0;
		ys=0;
	}
	public Lines(HashSet<DEdge> full,HashSet<DEdge> mst,double startx,double starty){
		super();
		full_edges=full;
		mst_edges=mst;
		xs=startx;
		ys=starty;
	}
	public void paint(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		g2.scale(50, 50);
		g2.setBackground(Color.white);
		float dash[] = { 0.1f };
		g2.setStroke(new BasicStroke(0.005f, BasicStroke.CAP_BUTT,
		        BasicStroke.JOIN_MITER, 10.0f, dash, 0.0f));
		g2.setColor(Color.gray);
		for(DEdge line:full_edges){
			Shape k = new Line2D.Double(line.p[0].x-xs,line.p[0].y-ys, line.p[1].x-xs,line.p[1].y-ys);
			g2.draw(k);
		}
		g2.setColor(Color.BLUE);
		g2.setStroke(new BasicStroke((float) 0.01));
		for(DEdge line:mst_edges){
			Shape k = new Line2D.Double(line.p[0].x-xs,line.p[0].y-ys, line.p[1].x-xs,line.p[1].y-ys);
			g2.draw(k);
		}
//		g2.setStroke(new BasicStroke(3));
//		for(DPoint point:p){
//			Shape k = new Line2D.Double(point.x,point.y, point.x,point.y);
//			g2.draw(k);
//		}
	}
}
