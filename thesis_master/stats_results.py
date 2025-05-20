"""
Results when using a terrain with both craters/slopes and rocks/obstacles. The weights are set such that the robot strictly avoids any kind of slope,
and therefore mainly drives on flat ground.
"""

# region With a 1000 trajectories, rocks present, and weight of 35.5 for slopes
print("\n \n Performance comparison with 1000 trajectories \n \n")  

"Distance critics:"
mppi2d = 157.0601466517287
mppi3d = 156.93851618362706
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Distance critics (after removing the five greatest elements):"
mppi2d = 154.53377377708279
mppi3d = 154.31781843329605
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 266.8662668001854
mppi3d = 266.02418155993445
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 258.18399895562067
mppi3d = 256.2657767401801
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 469.51098787986626
mppi3d = 467.8656440346928
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 459.70219082302515
mppi3d = 456.65676766854745
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Obstacle critics:"
print("Note that there were two collisions when using 2D, against none when using 3D")
mppi2d = 818.652329008458
mppi3d = 806.6215604815567
print("The obstacle avoidance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Obstacle critics (after removing the five greatest elements):"
mppi2d = 802.8696806165907
mppi3d = 788.898456133329
print("The obstacle avoidance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  
# endregion


# region With a 500 trajectories, rocks present, and weight of 35.5 for slopes

print("\n\n Performance comparison with 500 trajectories \n\n")  

"Distance critics:"
mppi2d = 156.33677286775622
mppi3d = 156.26482647974973
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  


"Distance critics (after removing the five greatest elements):"
mppi2d = 154.14648750399863
mppi3d = 153.5281862400031
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 488.1503110613142
mppi3d = 411.14561168964093
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 413.4460413315717
mppi3d = 397.98405034491356
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 497.32353918892994
mppi3d = 485.1042996920072
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 482.18962486117493
mppi3d = 471.6637053794049
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Obstacle critics:"
mppi2d = 831.8452251979282
mppi3d = 808.4707240324753
print("The obstacle avoidance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Obstacle critics (after removing the five greatest elements):"
mppi2d = 810.6370065726486
mppi3d = 786.6106551961695
print("The obstacle avoidance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion


# region With 350 trajectories, rocks present, and weight of 35.5 for slopes

print("\n\n Performance comparison with 350 trajectories \n\n")  

"Distance critics:"
mppi2d = 156.41957532076998
mppi3d = 156.98543431009713
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  


"Distance critics (after removing the five greatest elements):"
mppi2d = 152.9716844695662
mppi3d = 154.2227690767546
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 437.4137330136057
mppi3d = 431.6406299138473
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 416.2700732195819
mppi3d = 416.3383978384513
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 489.14283331509296
mppi3d = 483.26717777575476
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 477.5133615169885
mppi3d = 475.8472154405382
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Obstacle critics:"
mppi2d = 839.2649792170121
mppi3d = 814.762789871733
print("The obstacle avoidance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Obstacle critics (after removing the five greatest elements):"
mppi2d = 818.4910682395653
mppi3d = 793.8123896916708
print("The obstacle avoidance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion


"""
Results when using a terrain with craters/slopes only. The weights are set such that the robot strictly avoids any kind of slope,
and therefore mainly drives on flat ground.
"""


# region With 1000 trajectories, no rocks, and weight of 35.5 for slopes

print("\n\n Performance comparison with 1000 trajectories \n\n")  

"Distance critics:"
mppi2d = 144.31864041654768
mppi3d = 144.203315224062
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Distance critics (after removing the five greatest elements):"
mppi2d = 142.19372257262836
mppi3d = 142.15147176885446
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 148.34266882427667
mppi3d = 147.60379105907376
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 145.25512709441008
mppi3d = 143.8919431898329
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 397.71945707676775
mppi3d = 391.5870707883673
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 387.7189715350116
mppi3d = 384.6599217167607
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion


# region With 500 trajectories, no rocks, and weight of 35.5 for slopes


print("\n\n Performance comparison with 500 trajectories \n\n")  

"Distance critics:"
mppi2d = 146.53548660607692
mppi3d = 146.79641080815833
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Distance critics (after removing the five greatest elements):"
mppi2d = 144.56960351904195
mppi3d = 144.54971580578055
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 214.8674339682369
mppi3d = 213.33527619959943
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 209.73696192988643
mppi3d = 207.99131520589194
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 410.97989253674524
mppi3d = 412.1542415295617
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 402.3860857928241
mppi3d = 404.0745013201678
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion


# region With 350 trajectories, no rocks, and weight of 35.5 for slopes

print("\n\n Performance comparison with 350 trajectories \n\n")  

"Distance critics:"
mppi2d = 142.5813808740043
mppi3d = 142.93934474437722
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Distance critics (after removing the five greatest elements):"
mppi2d = 140.62370623187851
mppi3d = 140.99739595375465
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 235.91674442614539
mppi3d = 236.489162639036
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 227.96784916630497
mppi3d = 231.29786512586804
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 404.2259888729807
mppi3d =401.689697265625
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 396.9287567138672
mppi3d = 394.90848343460647
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion


"""
Results using a terrain with craters and rocks. The weights of the slope avoidance are reduced such that it drives more
on inclined and curved terrain.
"""

# region With 500 trajectories, rocks, and weight of 5.5 for slopes

print("\n \n 500 trajectories \n \n")  

"Distance critics:"
mppi2d = 152.8202643883053
mppi3d = 153.09367205548043
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Distance critics (after removing the five greatest elements):"
mppi2d = 150.49124918085082
mppi3d = 150.93969032568398
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 292.1753436589645
mppi3d = 278.4544814804853
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 281.2927053946036
mppi3d = 271.55989555076314
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 530.6042283914857
mppi3d = 528.4832039525953
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 519.7697550455729
mppi3d = 517.7562323676216
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Obstacle critics:"
mppi2d = 300.30377119678565 # 792.3805907297942
mppi3d = 288.63526994090967 # 781.657244666148
print("The obstacle avoidance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Obstacle critics (after removing the five greatest elements):"
mppi2d = 282.787557248716
mppi3d = 266.6440655743634
print("The obstacle avoidance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion

# region With 350 trajectories, rocks, and weight of 5.5 for slopes

print("\n \n 350 trajectories \n \n")  

"Distance critics:"
mppi2d = 152.8202643883053
mppi3d = 153.09367205548043
print("The distance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Distance critics (after removing the five greatest elements):"
mppi2d = 150.49124918085082
mppi3d = 150.93969032568398
print("The distance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Speed critics:"
mppi2d = 320.05381800764695
mppi3d = 308.31120946851826
print("The speed score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Speed critics (after removing the five greatest elements):"
mppi2d = 310.8985477023655
mppi3d = 301.30478526927806
print("The speed score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Slope critics:"
mppi2d = 552.6872144796081
mppi3d = 540.9001299324682
print("The slope score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Slope critics (after removing the five greatest elements):"
mppi2d = 539.1364531340422
mppi3d = 528.1480215567129
print("The slope score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

"Obstacle critics:"
mppi2d = 150.3898349697307 # 790.2195188877946
mppi3d = 146.17651237875728 # 755.5135478650109
print("The obstacle avoidance score is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "%")  

"Obstacle critics (after removing the five greatest elements):"
mppi2d = 136.14954496313024 # 772.30936827483
mppi3d = 130.76350092004847 # 733.853330400255
print("The obstacle avoidance score (filtered) is beaten by:", (mppi2d-mppi3d)*100/mppi3d, "% \n")  

# endregion
