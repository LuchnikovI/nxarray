#!/usr/bin/env python3

# fmt: off
# type: ignore
from numpy.random import normal
from numpy import isclose
from nxarray import NXArray

def test_autogen():
	x17 = NXArray(normal(size = (4,5,4,)), "i19","i20","i18",)
	x18 = NXArray(normal(size = (5,1,2,4,3,)), "i20","i17","i16","i19","i13",)
	x15 = x17 * x18
	assert set(x15.index_ids) == set(["i18","i13","i16","i17",])
	assert x15.release_array("i18","i13","i16","i17",).shape == (4,3,2,1,)
	assert x15.rank == 4
	x19 = NXArray(normal(size = (1,5,3,4,2,)), "i21","i15","i14","i18","i16",)
	x20 = NXArray(normal(size = (1,1,)), "i21","i17",)
	x16 = x19 * x20
	assert set(x16.index_ids) == set(["i15","i18","i16","i14","i17",])
	assert x16.release_array("i15","i18","i16","i14","i17",).shape == (5,4,2,3,1,)
	assert x16.rank == 5
	x13 = x15 * x16
	assert set(x13.index_ids) == set(["i13","i14","i15",])
	assert x13.release_array("i13","i14","i15",).shape == (3,3,5,)
	assert x13.rank == 3
	x23 = NXArray(normal(size = (4,3,5,4,3,)), "i12","i13","i22","i8","i14",)
	x24 = NXArray(normal(size = ()), )
	x21 = x23 * x24
	assert set(x21.index_ids) == set(["i13","i12","i22","i8","i14",])
	assert x21.release_array("i13","i12","i22","i8","i14",).shape == (3,4,5,4,3,)
	assert x21.rank == 5
	x25 = NXArray(normal(size = (1,1,2,)), "i23","i24","i25",)
	x26 = NXArray(normal(size = (1,5,5,1,2,)), "i23","i15","i22","i24","i25",)
	x22 = x25 * x26
	assert set(x22.index_ids) == set(["i15","i22",])
	assert x22.release_array("i15","i22",).shape == (5,5,)
	assert x22.rank == 2
	x14 = x21 * x22
	assert set(x14.index_ids) == set(["i8","i14","i13","i12","i15",])
	assert x14.release_array("i8","i14","i13","i12","i15",).shape == (4,3,3,4,5,)
	assert x14.rank == 5
	x11 = x13 * x14
	assert set(x11.index_ids) == set(["i12","i8",])
	assert x11.release_array("i12","i8",).shape == (4,4,)
	assert x11.rank == 2
	x31 = NXArray(normal(size = (5,5,)), "i31","i30",)
	x32 = NXArray(normal(size = (2,5,4,5,3,)), "i28","i30","i27","i31","i29",)
	x29 = x31 * x32
	assert set(x29.index_ids) == set(["i28","i29","i27",])
	assert x29.release_array("i28","i29","i27",).shape == (2,3,4,)
	assert x29.rank == 3
	x33 = NXArray(normal(size = (1,3,5,2,3,)), "i33","i32","i26","i9","i29",)
	x34 = NXArray(normal(size = (2,1,4,3,)), "i28","i33","i27","i32",)
	x30 = x33 * x34
	assert set(x30.index_ids) == set(["i26","i9","i29","i27","i28",])
	assert x30.release_array("i26","i9","i29","i27","i28",).shape == (5,2,3,4,2,)
	assert x30.rank == 5
	x27 = x29 * x30
	assert set(x27.index_ids) == set(["i9","i26",])
	assert x27.release_array("i9","i26",).shape == (2,5,)
	assert x27.rank == 2
	x37 = NXArray(normal(size = (2,)), "i36",)
	x38 = NXArray(normal(size = (2,5,1,2,5,)), "i10","i34","i11","i36","i35",)
	x35 = x37 * x38
	assert set(x35.index_ids) == set(["i11","i35","i34","i10",])
	assert x35.release_array("i11","i35","i34","i10",).shape == (1,5,5,2,)
	assert x35.rank == 4
	x39 = NXArray(normal(size = (2,4,3,2,)), "i37","i12","i5","i38",)
	x40 = NXArray(normal(size = (5,5,2,2,5,)), "i26","i34","i37","i38","i35",)
	x36 = x39 * x40
	assert set(x36.index_ids) == set(["i5","i12","i26","i34","i35",])
	assert x36.release_array("i5","i12","i26","i34","i35",).shape == (3,4,5,5,5,)
	assert x36.rank == 5
	x28 = x35 * x36
	assert set(x28.index_ids) == set(["i10","i11","i12","i26","i5",])
	assert x28.release_array("i10","i11","i12","i26","i5",).shape == (2,1,4,5,3,)
	assert x28.rank == 5
	x12 = x27 * x28
	assert set(x12.index_ids) == set(["i9","i10","i12","i11","i5",])
	assert x12.release_array("i9","i10","i12","i11","i5",).shape == (2,2,4,1,3,)
	assert x12.rank == 5
	x9 = x11 * x12
	assert set(x9.index_ids) == set(["i8","i10","i5","i11","i9",])
	assert x9.release_array("i8","i10","i5","i11","i9",).shape == (4,2,3,1,2,)
	assert x9.rank == 5
	x47 = NXArray(normal(size = (3,1,4,5,4,)), "i42","i43","i40","i39","i41",)
	x48 = NXArray(normal(size = ()), )
	x45 = x47 * x48
	assert set(x45.index_ids) == set(["i41","i39","i42","i43","i40",])
	assert x45.release_array("i41","i39","i42","i43","i40",).shape == (4,5,3,1,4,)
	assert x45.rank == 5
	x49 = NXArray(normal(size = (4,4,2,5,3,)), "i40","i41","i44","i39","i42",)
	x50 = NXArray(normal(size = (2,1,)), "i44","i43",)
	x46 = x49 * x50
	assert set(x46.index_ids) == set(["i42","i39","i41","i40","i43",])
	assert x46.release_array("i42","i39","i41","i40","i43",).shape == (3,5,4,4,1,)
	assert x46.rank == 5
	x43 = x45 * x46
	assert set(x43.index_ids) == set([])
	assert x43.release_array().shape == ()
	assert x43.rank == 0
	x53 = NXArray(normal(size = (3,2,4,1,)), "i48","i9","i47","i11",)
	x54 = NXArray(normal(size = (4,4,1,3,4,)), "i47","i8","i46","i48","i45",)
	x51 = x53 * x54
	assert set(x51.index_ids) == set(["i11","i9","i46","i45","i8",])
	assert x51.release_array("i11","i9","i46","i45","i8",).shape == (1,2,1,4,4,)
	assert x51.rank == 5
	x55 = NXArray(normal(size = (1,1,2,4,3,)), "i49","i46","i10","i45","i7",)
	x56 = NXArray(normal(size = (1,)), "i49",)
	x52 = x55 * x56
	assert set(x52.index_ids) == set(["i45","i46","i10","i7",])
	assert x52.release_array("i45","i46","i10","i7",).shape == (4,1,2,3,)
	assert x52.rank == 4
	x44 = x51 * x52
	assert set(x44.index_ids) == set(["i9","i8","i11","i10","i7",])
	assert x44.release_array("i9","i8","i11","i10","i7",).shape == (2,4,1,2,3,)
	assert x44.rank == 5
	x41 = x43 * x44
	assert set(x41.index_ids) == set(["i8","i10","i9","i7","i11",])
	assert x41.release_array("i8","i10","i9","i7","i11",).shape == (4,2,2,3,1,)
	assert x41.rank == 5
	x61 = NXArray(normal(size = (4,1,3,)), "i58","i56","i57",)
	x62 = NXArray(normal(size = (5,1,3,4,5,)), "i50","i56","i57","i58","i55",)
	x59 = x61 * x62
	assert set(x59.index_ids) == set(["i55","i50",])
	assert x59.release_array("i55","i50",).shape == (5,5,)
	assert x59.rank == 2
	x63 = NXArray(normal(size = (1,5,1,5,5,)), "i59","i53","i51","i54","i55",)
	x64 = NXArray(normal(size = (1,1,)), "i59","i52",)
	x60 = x63 * x64
	assert set(x60.index_ids) == set(["i54","i51","i55","i53","i52",])
	assert x60.release_array("i54","i51","i55","i53","i52",).shape == (5,1,5,5,1,)
	assert x60.rank == 5
	x57 = x59 * x60
	assert set(x57.index_ids) == set(["i50","i54","i51","i52","i53",])
	assert x57.release_array("i50","i54","i51","i52","i53",).shape == (5,5,1,1,5,)
	assert x57.rank == 5
	x67 = NXArray(normal(size = (1,5,4,5,5,)), "i62","i60","i61","i50","i53",)
	x68 = NXArray(normal(size = (1,)), "i62",)
	x65 = x67 * x68
	assert set(x65.index_ids) == set(["i60","i50","i53","i61",])
	assert x65.release_array("i60","i50","i53","i61",).shape == (5,5,5,4,)
	assert x65.rank == 4
	x69 = NXArray(normal(size = (5,4,5,4,5,)), "i64","i63","i60","i61","i54",)
	x70 = NXArray(normal(size = (5,1,1,4,)), "i64","i52","i51","i63",)
	x66 = x69 * x70
	assert set(x66.index_ids) == set(["i60","i61","i54","i51","i52",])
	assert x66.release_array("i60","i61","i54","i51","i52",).shape == (5,4,5,1,1,)
	assert x66.rank == 5
	x58 = x65 * x66
	assert set(x58.index_ids) == set(["i50","i53","i51","i52","i54",])
	assert x58.release_array("i50","i53","i51","i52","i54",).shape == (5,5,1,1,5,)
	assert x58.rank == 5
	x42 = x57 * x58
	assert set(x42.index_ids) == set([])
	assert x42.release_array().shape == ()
	assert x42.rank == 0
	x10 = x41 * x42
	assert set(x10.index_ids) == set(["i8","i11","i9","i10","i7",])
	assert x10.release_array("i8","i11","i9","i10","i7",).shape == (4,1,2,2,3,)
	assert x10.rank == 5
	x7 = x9 * x10
	assert set(x7.index_ids) == set(["i5","i7",])
	assert x7.release_array("i5","i7",).shape == (3,3,)
	assert x7.rank == 2
	x79 = NXArray(normal(size = ()), )
	x80 = NXArray(normal(size = (3,2,4,4,4,)), "i2","i1","i70","i67","i69",)
	x77 = x79 * x80
	assert set(x77.index_ids) == set(["i67","i2","i69","i70","i1",])
	assert x77.release_array("i67","i2","i69","i70","i1",).shape == (4,3,4,4,2,)
	assert x77.rank == 5
	x81 = NXArray(normal(size = (4,1,4,4,4,)), "i70","i68","i69","i71","i66",)
	x82 = NXArray(normal(size = (4,)), "i71",)
	x78 = x81 * x82
	assert set(x78.index_ids) == set(["i66","i70","i68","i69",])
	assert x78.release_array("i66","i70","i68","i69",).shape == (4,4,1,4,)
	assert x78.rank == 4
	x75 = x77 * x78
	assert set(x75.index_ids) == set(["i1","i2","i67","i66","i68",])
	assert x75.release_array("i1","i2","i67","i66","i68",).shape == (2,3,4,4,1,)
	assert x75.rank == 5
	x85 = NXArray(normal(size = (1,2,1,3,)), "i76","i73","i74","i75",)
	x86 = NXArray(normal(size = (3,2,1,1,3,)), "i72","i73","i76","i74","i75",)
	x83 = x85 * x86
	assert set(x83.index_ids) == set(["i72",])
	assert x83.release_array("i72",).shape == (3,)
	assert x83.rank == 1
	x87 = NXArray(normal(size = (4,1,3,3,3,)), "i67","i68","i77","i72","i78",)
	x88 = NXArray(normal(size = (3,5,3,3,)), "i78","i65","i77","i7",)
	x84 = x87 * x88
	assert set(x84.index_ids) == set(["i72","i68","i67","i65","i7",])
	assert x84.release_array("i72","i68","i67","i65","i7",).shape == (3,1,4,5,3,)
	assert x84.rank == 5
	x76 = x83 * x84
	assert set(x76.index_ids) == set(["i67","i68","i65","i7",])
	assert x76.release_array("i67","i68","i65","i7",).shape == (4,1,5,3,)
	assert x76.rank == 4
	x73 = x75 * x76
	assert set(x73.index_ids) == set(["i66","i1","i2","i65","i7",])
	assert x73.release_array("i66","i1","i2","i65","i7",).shape == (4,2,3,5,3,)
	assert x73.rank == 5
	x93 = NXArray(normal(size = (2,2,4,5,5,)), "i82","i80","i79","i83","i81",)
	x94 = NXArray(normal(size = ()), )
	x91 = x93 * x94
	assert set(x91.index_ids) == set(["i80","i83","i82","i81","i79",])
	assert x91.release_array("i80","i83","i82","i81","i79",).shape == (2,5,2,5,4,)
	assert x91.rank == 5
	x95 = NXArray(normal(size = (2,5,2,2,4,)), "i87","i85","i84","i86","i88",)
	x96 = NXArray(normal(size = (2,5,2,2,4,)), "i87","i85","i86","i84","i88",)
	x92 = x95 * x96
	assert set(x92.index_ids) == set([])
	assert x92.release_array().shape == ()
	assert x92.rank == 0
	x89 = x91 * x92
	assert set(x89.index_ids) == set(["i82","i83","i81","i79","i80",])
	assert x89.release_array("i82","i83","i81","i79","i80",).shape == (2,5,5,4,2,)
	assert x89.rank == 5
	x99 = NXArray(normal(size = (3,4,5,1,5,)), "i92","i89","i90","i93","i91",)
	x100 = NXArray(normal(size = (3,4,5,5,1,)), "i92","i89","i90","i91","i93",)
	x97 = x99 * x100
	assert set(x97.index_ids) == set([])
	assert x97.release_array().shape == ()
	assert x97.rank == 0
	x101 = NXArray(normal(size = (3,2,5,4,2,)), "i94","i95","i81","i79","i80",)
	x102 = NXArray(normal(size = (5,3,2,2,)), "i83","i94","i82","i95",)
	x98 = x101 * x102
	assert set(x98.index_ids) == set(["i81","i80","i79","i82","i83",])
	assert x98.release_array("i81","i80","i79","i82","i83",).shape == (5,2,4,2,5,)
	assert x98.rank == 5
	x90 = x97 * x98
	assert set(x90.index_ids) == set(["i81","i79","i82","i80","i83",])
	assert x90.release_array("i81","i79","i82","i80","i83",).shape == (5,4,2,2,5,)
	assert x90.rank == 5
	x74 = x89 * x90
	assert set(x74.index_ids) == set([])
	assert x74.release_array().shape == ()
	assert x74.rank == 0
	x71 = x73 * x74
	assert set(x71.index_ids) == set(["i7","i2","i1","i66","i65",])
	assert x71.release_array("i7","i2","i1","i66","i65",).shape == (3,3,2,4,5,)
	assert x71.rank == 5
	x109 = NXArray(normal(size = (3,1,3,2,4,)), "i103","i101","i102","i100","i104",)
	x110 = NXArray(normal(size = (1,3,2,3,4,)), "i101","i103","i100","i102","i104",)
	x107 = x109 * x110
	assert set(x107.index_ids) == set([])
	assert x107.release_array().shape == ()
	assert x107.rank == 0
	x111 = NXArray(normal(size = ()), )
	x112 = NXArray(normal(size = (2,5,2,4,3,)), "i97","i65","i98","i99","i96",)
	x108 = x111 * x112
	assert set(x108.index_ids) == set(["i97","i98","i99","i65","i96",])
	assert x108.release_array("i97","i98","i99","i65","i96",).shape == (2,2,4,5,3,)
	assert x108.rank == 5
	x105 = x107 * x108
	assert set(x105.index_ids) == set(["i98","i99","i97","i65","i96",])
	assert x105.release_array("i98","i99","i97","i65","i96",).shape == (2,4,2,5,3,)
	assert x105.rank == 5
	x115 = NXArray(normal(size = (3,2,4,1,3,)), "i106","i105","i110","i109","i107",)
	x116 = NXArray(normal(size = (1,4,4,4,)), "i109","i110","i108","i99",)
	x113 = x115 * x116
	assert set(x113.index_ids) == set(["i107","i105","i106","i108","i99",])
	assert x113.release_array("i107","i105","i106","i108","i99",).shape == (3,2,3,4,4,)
	assert x113.rank == 5
	x117 = NXArray(normal(size = (2,3,3,1,2,)), "i98","i111","i106","i112","i105",)
	x118 = NXArray(normal(size = (4,3,3,1,)), "i108","i107","i111","i112",)
	x114 = x117 * x118
	assert set(x114.index_ids) == set(["i105","i106","i98","i108","i107",])
	assert x114.release_array("i105","i106","i98","i108","i107",).shape == (2,3,2,4,3,)
	assert x114.rank == 5
	x106 = x113 * x114
	assert set(x106.index_ids) == set(["i99","i98",])
	assert x106.release_array("i99","i98",).shape == (4,2,)
	assert x106.rank == 2
	x103 = x105 * x106
	assert set(x103.index_ids) == set(["i97","i96","i65",])
	assert x103.release_array("i97","i96","i65",).shape == (2,3,5,)
	assert x103.rank == 3
	x123 = NXArray(normal(size = ()), )
	x124 = NXArray(normal(size = (5,4,3,2,4,)), "i113","i114","i115","i117","i116",)
	x121 = x123 * x124
	assert set(x121.index_ids) == set(["i115","i116","i117","i114","i113",])
	assert x121.release_array("i115","i116","i117","i114","i113",).shape == (3,4,2,4,5,)
	assert x121.rank == 5
	x125 = NXArray(normal(size = (2,4,4,5,2,)), "i118","i114","i116","i113","i117",)
	x126 = NXArray(normal(size = (2,3,)), "i118","i115",)
	x122 = x125 * x126
	assert set(x122.index_ids) == set(["i116","i117","i113","i114","i115",])
	assert x122.release_array("i116","i117","i113","i114","i115",).shape == (4,2,5,4,3,)
	assert x122.rank == 5
	x119 = x121 * x122
	assert set(x119.index_ids) == set([])
	assert x119.release_array().shape == ()
	assert x119.rank == 0
	x129 = NXArray(normal(size = (3,4,5,2,2,)), "i96","i6","i121","i122","i120",)
	x130 = NXArray(normal(size = (3,5,2,)), "i119","i121","i122",)
	x127 = x129 * x130
	assert set(x127.index_ids) == set(["i120","i96","i6","i119",])
	assert x127.release_array("i120","i96","i6","i119",).shape == (2,3,4,3,)
	assert x127.rank == 4
	x131 = NXArray(normal(size = (4,2,2,2,)), "i124","i120","i123","i97",)
	x132 = NXArray(normal(size = (2,4,4,3,3,)), "i123","i124","i66","i0","i119",)
	x128 = x131 * x132
	assert set(x128.index_ids) == set(["i97","i120","i66","i119","i0",])
	assert x128.release_array("i97","i120","i66","i119","i0",).shape == (2,2,4,3,3,)
	assert x128.rank == 5
	x120 = x127 * x128
	assert set(x120.index_ids) == set(["i6","i96","i97","i66","i0",])
	assert x120.release_array("i6","i96","i97","i66","i0",).shape == (4,3,2,4,3,)
	assert x120.rank == 5
	x104 = x119 * x120
	assert set(x104.index_ids) == set(["i6","i96","i97","i66","i0",])
	assert x104.release_array("i6","i96","i97","i66","i0",).shape == (4,3,2,4,3,)
	assert x104.rank == 5
	x72 = x103 * x104
	assert set(x72.index_ids) == set(["i65","i6","i66","i0",])
	assert x72.release_array("i65","i6","i66","i0",).shape == (5,4,4,3,)
	assert x72.rank == 4
	x8 = x71 * x72
	assert set(x8.index_ids) == set(["i7","i1","i2","i6","i0",])
	assert x8.release_array("i7","i1","i2","i6","i0",).shape == (3,2,3,4,3,)
	assert x8.rank == 5
	x5 = x7 * x8
	assert set(x5.index_ids) == set(["i5","i6","i0","i1","i2",])
	assert x5.release_array("i5","i6","i0","i1","i2",).shape == (3,4,3,2,3,)
	assert x5.rank == 5
	x143 = NXArray(normal(size = (4,2,)), "i137","i134",)
	x144 = NXArray(normal(size = (2,5,5,4,5,)), "i133","i135","i132","i137","i136",)
	x141 = x143 * x144
	assert set(x141.index_ids) == set(["i134","i136","i133","i132","i135",])
	assert x141.release_array("i134","i136","i133","i132","i135",).shape == (2,5,2,5,5,)
	assert x141.rank == 5
	x145 = NXArray(normal(size = (1,3,5,4,2,)), "i138","i139","i142","i141","i140",)
	x146 = NXArray(normal(size = (2,1,4,5,3,)), "i140","i138","i141","i142","i139",)
	x142 = x145 * x146
	assert set(x142.index_ids) == set([])
	assert x142.release_array().shape == ()
	assert x142.rank == 0
	x139 = x141 * x142
	assert set(x139.index_ids) == set(["i134","i136","i132","i135","i133",])
	assert x139.release_array("i134","i136","i132","i135","i133",).shape == (2,5,5,5,2,)
	assert x139.rank == 5
	x149 = NXArray(normal(size = (4,5,2,4,5,)), "i143","i135","i134","i144","i136",)
	x150 = NXArray(normal(size = ()), )
	x147 = x149 * x150
	assert set(x147.index_ids) == set(["i143","i135","i134","i136","i144",])
	assert x147.release_array("i143","i135","i134","i136","i144",).shape == (4,5,2,5,4,)
	assert x147.rank == 5
	x151 = NXArray(normal(size = (5,2,1,4,4,)), "i147","i145","i146","i130","i143",)
	x152 = NXArray(normal(size = (4,5,1,2,)), "i144","i147","i146","i145",)
	x148 = x151 * x152
	assert set(x148.index_ids) == set(["i130","i143","i144",])
	assert x148.release_array("i130","i143","i144",).shape == (4,4,4,)
	assert x148.rank == 3
	x140 = x147 * x148
	assert set(x140.index_ids) == set(["i134","i136","i135","i130",])
	assert x140.release_array("i134","i136","i135","i130",).shape == (2,5,5,4,)
	assert x140.rank == 4
	x137 = x139 * x140
	assert set(x137.index_ids) == set(["i132","i133","i130",])
	assert x137.release_array("i132","i133","i130",).shape == (5,2,4,)
	assert x137.rank == 3
	x157 = NXArray(normal(size = (3,5,3,3,2,)), "i151","i150","i154","i152","i153",)
	x158 = NXArray(normal(size = (3,5,2,3,3,)), "i154","i150","i153","i151","i152",)
	x155 = x157 * x158
	assert set(x155.index_ids) == set([])
	assert x155.release_array().shape == ()
	assert x155.rank == 0
	x159 = NXArray(normal(size = ()), )
	x160 = NXArray(normal(size = (1,5,2,3,2,)), "i149","i132","i128","i127","i148",)
	x156 = x159 * x160
	assert set(x156.index_ids) == set(["i149","i132","i148","i127","i128",])
	assert x156.release_array("i149","i132","i148","i127","i128",).shape == (1,5,2,3,2,)
	assert x156.rank == 5
	x153 = x155 * x156
	assert set(x153.index_ids) == set(["i132","i148","i127","i149","i128",])
	assert x153.release_array("i132","i148","i127","i149","i128",).shape == (5,2,3,1,2,)
	assert x153.rank == 5
	x163 = NXArray(normal(size = (1,2,4,4,1,)), "i149","i148","i158","i156","i155",)
	x164 = NXArray(normal(size = (4,2,)), "i158","i157",)
	x161 = x163 * x164
	assert set(x161.index_ids) == set(["i156","i149","i155","i148","i157",])
	assert x161.release_array("i156","i149","i155","i148","i157",).shape == (4,1,1,2,2,)
	assert x161.rank == 5
	x165 = NXArray(normal(size = (2,1,2,1,4,)), "i157","i159","i131","i155","i156",)
	x166 = NXArray(normal(size = (1,2,)), "i159","i133",)
	x162 = x165 * x166
	assert set(x162.index_ids) == set(["i155","i156","i131","i157","i133",])
	assert x162.release_array("i155","i156","i131","i157","i133",).shape == (1,4,2,2,2,)
	assert x162.rank == 5
	x154 = x161 * x162
	assert set(x154.index_ids) == set(["i148","i149","i131","i133",])
	assert x154.release_array("i148","i149","i131","i133",).shape == (2,1,2,2,)
	assert x154.rank == 4
	x138 = x153 * x154
	assert set(x138.index_ids) == set(["i128","i127","i132","i131","i133",])
	assert x138.release_array("i128","i127","i132","i131","i133",).shape == (2,3,5,2,2,)
	assert x138.rank == 5
	x135 = x137 * x138
	assert set(x135.index_ids) == set(["i130","i127","i128","i131",])
	assert x135.release_array("i130","i127","i128","i131",).shape == (4,3,2,2,)
	assert x135.rank == 4
	x173 = NXArray(normal(size = (2,2,3,4,3,)), "i163","i161","i166","i164","i167",)
	x174 = NXArray(normal(size = (3,3,1,3,)), "i167","i166","i165","i162",)
	x171 = x173 * x174
	assert set(x171.index_ids) == set(["i163","i164","i161","i165","i162",])
	assert x171.release_array("i163","i164","i161","i165","i162",).shape == (2,4,2,1,3,)
	assert x171.rank == 5
	x175 = NXArray(normal(size = (3,4,3,5,4,)), "i169","i168","i170","i171","i164",)
	x176 = NXArray(normal(size = (4,3,1,5,3,)), "i168","i170","i165","i171","i169",)
	x172 = x175 * x176
	assert set(x172.index_ids) == set(["i164","i165",])
	assert x172.release_array("i164","i165",).shape == (4,1,)
	assert x172.rank == 2
	x169 = x171 * x172
	assert set(x169.index_ids) == set(["i162","i161","i163",])
	assert x169.release_array("i162","i161","i163",).shape == (3,2,2,)
	assert x169.rank == 3
	x179 = NXArray(normal(size = (4,2,2,3,4,)), "i172","i163","i161","i162","i126",)
	x180 = NXArray(normal(size = ()), )
	x177 = x179 * x180
	assert set(x177.index_ids) == set(["i162","i163","i126","i172","i161",])
	assert x177.release_array("i162","i163","i126","i172","i161",).shape == (3,2,4,4,2,)
	assert x177.rank == 5
	x181 = NXArray(normal(size = (5,4,3,)), "i173","i174","i175",)
	x182 = NXArray(normal(size = (5,3,4,4,5,)), "i160","i175","i174","i172","i173",)
	x178 = x181 * x182
	assert set(x178.index_ids) == set(["i172","i160",])
	assert x178.release_array("i172","i160",).shape == (4,5,)
	assert x178.rank == 2
	x170 = x177 * x178
	assert set(x170.index_ids) == set(["i163","i126","i161","i162","i160",])
	assert x170.release_array("i163","i126","i161","i162","i160",).shape == (2,4,2,3,5,)
	assert x170.rank == 5
	x167 = x169 * x170
	assert set(x167.index_ids) == set(["i160","i126",])
	assert x167.release_array("i160","i126",).shape == (5,4,)
	assert x167.rank == 2
	x187 = NXArray(normal(size = (2,4,4,3,5,)), "i178","i130","i180","i177","i179",)
	x188 = NXArray(normal(size = (2,5,4,)), "i178","i179","i180",)
	x185 = x187 * x188
	assert set(x185.index_ids) == set(["i177","i130",])
	assert x185.release_array("i177","i130",).shape == (3,4,)
	assert x185.rank == 2
	x189 = NXArray(normal(size = (4,2,3,3,5,)), "i181","i182","i177","i176","i160",)
	x190 = NXArray(normal(size = (2,2,3,4,)), "i131","i182","i129","i181",)
	x186 = x189 * x190
	assert set(x186.index_ids) == set(["i177","i160","i176","i129","i131",])
	assert x186.release_array("i177","i160","i176","i129","i131",).shape == (3,5,3,3,2,)
	assert x186.rank == 5
	x183 = x185 * x186
	assert set(x183.index_ids) == set(["i130","i160","i131","i129","i176",])
	assert x183.release_array("i130","i160","i131","i129","i176",).shape == (4,5,2,3,3,)
	assert x183.rank == 5
	x193 = NXArray(normal(size = (1,2,)), "i186","i187",)
	x194 = NXArray(normal(size = (1,2,1,5,1,)), "i184","i187","i186","i183","i185",)
	x191 = x193 * x194
	assert set(x191.index_ids) == set(["i184","i183","i185",])
	assert x191.release_array("i184","i183","i185",).shape == (1,5,1,)
	assert x191.rank == 3
	x195 = NXArray(normal(size = (2,1,)), "i188","i185",)
	x196 = NXArray(normal(size = (4,5,3,2,1,)), "i125","i183","i176","i188","i184",)
	x192 = x195 * x196
	assert set(x192.index_ids) == set(["i185","i176","i183","i125","i184",])
	assert x192.release_array("i185","i176","i183","i125","i184",).shape == (1,3,5,4,1,)
	assert x192.rank == 5
	x184 = x191 * x192
	assert set(x184.index_ids) == set(["i176","i125",])
	assert x184.release_array("i176","i125",).shape == (3,4,)
	assert x184.rank == 2
	x168 = x183 * x184
	assert set(x168.index_ids) == set(["i160","i129","i130","i131","i125",])
	assert x168.release_array("i160","i129","i130","i131","i125",).shape == (5,3,4,2,4,)
	assert x168.rank == 5
	x136 = x167 * x168
	assert set(x136.index_ids) == set(["i126","i130","i131","i129","i125",])
	assert x136.release_array("i126","i130","i131","i129","i125",).shape == (4,4,2,3,4,)
	assert x136.rank == 5
	x133 = x135 * x136
	assert set(x133.index_ids) == set(["i127","i128","i126","i129","i125",])
	assert x133.release_array("i127","i128","i126","i129","i125",).shape == (3,2,4,3,4,)
	assert x133.rank == 5
	x205 = NXArray(normal(size = (1,1,3,3,5,)), "i194","i190","i195","i196","i193",)
	x206 = NXArray(normal(size = (3,1,3,)), "i196","i194","i195",)
	x203 = x205 * x206
	assert set(x203.index_ids) == set(["i193","i190",])
	assert x203.release_array("i193","i190",).shape == (5,1,)
	assert x203.rank == 2
	x207 = NXArray(normal(size = (1,1,5,3,2,)), "i192","i191","i193","i197","i128",)
	x208 = NXArray(normal(size = (3,5,)), "i197","i189",)
	x204 = x207 * x208
	assert set(x204.index_ids) == set(["i192","i193","i128","i191","i189",])
	assert x204.release_array("i192","i193","i128","i191","i189",).shape == (1,5,2,1,5,)
	assert x204.rank == 5
	x201 = x203 * x204
	assert set(x201.index_ids) == set(["i190","i191","i128","i189","i192",])
	assert x201.release_array("i190","i191","i128","i189","i192",).shape == (1,1,2,5,1,)
	assert x201.rank == 5
	x211 = NXArray(normal(size = (5,1,3,1,5,)), "i199","i203","i201","i198","i202",)
	x212 = NXArray(normal(size = (1,1,)), "i200","i203",)
	x209 = x211 * x212
	assert set(x209.index_ids) == set(["i201","i199","i198","i202","i200",])
	assert x209.release_array("i201","i199","i198","i202","i200",).shape == (3,5,1,5,1,)
	assert x209.rank == 5
	x213 = NXArray(normal(size = (5,3,1,1,5,)), "i199","i201","i198","i200","i202",)
	x214 = NXArray(normal(size = ()), )
	x210 = x213 * x214
	assert set(x210.index_ids) == set(["i198","i199","i202","i201","i200",])
	assert x210.release_array("i198","i199","i202","i201","i200",).shape == (1,5,5,3,1,)
	assert x210.rank == 5
	x202 = x209 * x210
	assert set(x202.index_ids) == set([])
	assert x202.release_array().shape == ()
	assert x202.rank == 0
	x199 = x201 * x202
	assert set(x199.index_ids) == set(["i192","i128","i189","i190","i191",])
	assert x199.release_array("i192","i128","i189","i190","i191",).shape == (1,2,5,1,1,)
	assert x199.rank == 5
	x219 = NXArray(normal(size = (3,2,2,5,2,)), "i209","i211","i208","i207","i210",)
	x220 = NXArray(normal(size = (5,2,2,3,2,)), "i207","i211","i210","i209","i208",)
	x217 = x219 * x220
	assert set(x217.index_ids) == set([])
	assert x217.release_array().shape == ()
	assert x217.rank == 0
	x221 = NXArray(normal(size = (3,1,5,1,4,)), "i212","i191","i213","i206","i204",)
	x222 = NXArray(normal(size = (1,3,5,3,)), "i190","i212","i213","i205",)
	x218 = x221 * x222
	assert set(x218.index_ids) == set(["i206","i204","i191","i190","i205",])
	assert x218.release_array("i206","i204","i191","i190","i205",).shape == (1,4,1,1,3,)
	assert x218.rank == 5
	x215 = x217 * x218
	assert set(x215.index_ids) == set(["i191","i190","i205","i204","i206",])
	assert x215.release_array("i191","i190","i205","i204","i206",).shape == (1,1,3,4,1,)
	assert x215.rank == 5
	x225 = NXArray(normal(size = (5,1,4,4,1,)), "i217","i192","i204","i216","i206",)
	x226 = NXArray(normal(size = (4,5,5,2,)), "i216","i217","i215","i214",)
	x223 = x225 * x226
	assert set(x223.index_ids) == set(["i192","i206","i204","i214","i215",])
	assert x223.release_array("i192","i206","i204","i214","i215",).shape == (1,1,4,2,5,)
	assert x223.rank == 5
	x227 = NXArray(normal(size = (1,2,5,5,)), "i220","i218","i219","i215",)
	x228 = NXArray(normal(size = (1,2,2,5,3,)), "i220","i218","i214","i219","i205",)
	x224 = x227 * x228
	assert set(x224.index_ids) == set(["i215","i205","i214",])
	assert x224.release_array("i215","i205","i214",).shape == (5,3,2,)
	assert x224.rank == 3
	x216 = x223 * x224
	assert set(x216.index_ids) == set(["i204","i192","i206","i205",])
	assert x216.release_array("i204","i192","i206","i205",).shape == (4,1,1,3,)
	assert x216.rank == 4
	x200 = x215 * x216
	assert set(x200.index_ids) == set(["i191","i190","i192",])
	assert x200.release_array("i191","i190","i192",).shape == (1,1,1,)
	assert x200.rank == 3
	x197 = x199 * x200
	assert set(x197.index_ids) == set(["i128","i189",])
	assert x197.release_array("i128","i189",).shape == (2,5,)
	assert x197.rank == 2
	x235 = NXArray(normal(size = (3,5,5,3,1,)), "i228","i223","i227","i226","i225",)
	x236 = NXArray(normal(size = (5,3,)), "i224","i228",)
	x233 = x235 * x236
	assert set(x233.index_ids) == set(["i223","i227","i226","i225","i224",])
	assert x233.release_array("i223","i227","i226","i225","i224",).shape == (5,5,3,1,5,)
	assert x233.rank == 5
	x237 = NXArray(normal(size = (5,3,5,1,5,)), "i223","i226","i227","i225","i224",)
	x238 = NXArray(normal(size = ()), )
	x234 = x237 * x238
	assert set(x234.index_ids) == set(["i224","i223","i225","i226","i227",])
	assert x234.release_array("i224","i223","i225","i226","i227",).shape == (5,5,1,3,5,)
	assert x234.rank == 5
	x231 = x233 * x234
	assert set(x231.index_ids) == set([])
	assert x231.release_array().shape == ()
	assert x231.rank == 0
	x241 = NXArray(normal(size = (3,2,)), "i222","i229",)
	x242 = NXArray(normal(size = (3,2,2,4,4,)), "i129","i221","i229","i125","i126",)
	x239 = x241 * x242
	assert set(x239.index_ids) == set(["i222","i126","i125","i129","i221",])
	assert x239.release_array("i222","i126","i125","i129","i221",).shape == (3,4,4,3,2,)
	assert x239.rank == 5
	x243 = NXArray(normal(size = (2,4,3,3,1,)), "i230","i231","i233","i234","i232",)
	x244 = NXArray(normal(size = (3,2,4,1,3,)), "i234","i230","i231","i232","i233",)
	x240 = x243 * x244
	assert set(x240.index_ids) == set([])
	assert x240.release_array().shape == ()
	assert x240.rank == 0
	x232 = x239 * x240
	assert set(x232.index_ids) == set(["i129","i125","i222","i221","i126",])
	assert x232.release_array("i129","i125","i222","i221","i126",).shape == (3,4,3,2,4,)
	assert x232.rank == 5
	x229 = x231 * x232
	assert set(x229.index_ids) == set(["i125","i222","i221","i129","i126",])
	assert x229.release_array("i125","i222","i221","i129","i126",).shape == (4,3,2,3,4,)
	assert x229.rank == 5
	x249 = NXArray(normal(size = (5,4,2,1,1,)), "i237","i241","i239","i236","i240",)
	x250 = NXArray(normal(size = (1,1,4,)), "i238","i240","i241",)
	x247 = x249 * x250
	assert set(x247.index_ids) == set(["i239","i236","i237","i238",])
	assert x247.release_array("i239","i236","i237","i238",).shape == (2,1,5,1,)
	assert x247.rank == 4
	x251 = NXArray(normal(size = (2,4,3,1,2,)), "i243","i242","i235","i236","i239",)
	x252 = NXArray(normal(size = (2,5,1,4,)), "i243","i237","i238","i242",)
	x248 = x251 * x252
	assert set(x248.index_ids) == set(["i236","i239","i235","i238","i237",])
	assert x248.release_array("i236","i239","i235","i238","i237",).shape == (1,2,3,1,5,)
	assert x248.rank == 5
	x245 = x247 * x248
	assert set(x245.index_ids) == set(["i235",])
	assert x245.release_array("i235",).shape == (3,)
	assert x245.rank == 1
	x255 = NXArray(normal(size = ()), )
	x256 = NXArray(normal(size = (3,3,3,1,2,)), "i127","i222","i235","i244","i221",)
	x253 = x255 * x256
	assert set(x253.index_ids) == set(["i244","i235","i127","i222","i221",])
	assert x253.release_array("i244","i235","i127","i222","i221",).shape == (1,3,3,3,2,)
	assert x253.rank == 5
	x257 = NXArray(normal(size = (1,1,3,4,1,)), "i247","i244","i245","i246","i248",)
	x258 = NXArray(normal(size = (3,4,1,5,1,)), "i245","i246","i247","i189","i248",)
	x254 = x257 * x258
	assert set(x254.index_ids) == set(["i244","i189",])
	assert x254.release_array("i244","i189",).shape == (1,5,)
	assert x254.rank == 2
	x246 = x253 * x254
	assert set(x246.index_ids) == set(["i235","i222","i221","i127","i189",])
	assert x246.release_array("i235","i222","i221","i127","i189",).shape == (3,3,2,3,5,)
	assert x246.rank == 5
	x230 = x245 * x246
	assert set(x230.index_ids) == set(["i222","i221","i127","i189",])
	assert x230.release_array("i222","i221","i127","i189",).shape == (3,2,3,5,)
	assert x230.rank == 4
	x198 = x229 * x230
	assert set(x198.index_ids) == set(["i129","i126","i125","i189","i127",])
	assert x198.release_array("i129","i126","i125","i189","i127",).shape == (3,4,4,5,3,)
	assert x198.rank == 5
	x134 = x197 * x198
	assert set(x134.index_ids) == set(["i128","i125","i129","i126","i127",])
	assert x134.release_array("i128","i125","i129","i126","i127",).shape == (2,4,3,4,3,)
	assert x134.rank == 5
	x6 = x133 * x134
	assert set(x6.index_ids) == set([])
	assert x6.release_array().shape == ()
	assert x6.rank == 0
	x3 = x5 * x6
	assert set(x3.index_ids) == set(["i2","i1","i0","i5","i6",])
	assert x3.release_array("i2","i1","i0","i5","i6",).shape == (3,2,3,3,4,)
	assert x3.rank == 5
	x271 = NXArray(normal(size = ()), )
	x272 = NXArray(normal(size = (5,5,2,1,5,)), "i260","i264","i261","i262","i263",)
	x269 = x271 * x272
	assert set(x269.index_ids) == set(["i260","i263","i262","i264","i261",])
	assert x269.release_array("i260","i263","i262","i264","i261",).shape == (5,5,1,5,2,)
	assert x269.rank == 5
	x273 = NXArray(normal(size = (5,5,5,2,1,)), "i263","i264","i265","i266","i267",)
	x274 = NXArray(normal(size = (2,5,1,)), "i266","i265","i267",)
	x270 = x273 * x274
	assert set(x270.index_ids) == set(["i264","i263",])
	assert x270.release_array("i264","i263",).shape == (5,5,)
	assert x270.rank == 2
	x267 = x269 * x270
	assert set(x267.index_ids) == set(["i261","i262","i260",])
	assert x267.release_array("i261","i262","i260",).shape == (2,1,5,)
	assert x267.rank == 3
	x277 = NXArray(normal(size = (1,3,5,5,5,)), "i268","i270","i272","i269","i271",)
	x278 = NXArray(normal(size = (3,5,5,1,5,)), "i270","i271","i272","i268","i269",)
	x275 = x277 * x278
	assert set(x275.index_ids) == set([])
	assert x275.release_array().shape == ()
	assert x275.rank == 0
	x279 = NXArray(normal(size = ()), )
	x280 = NXArray(normal(size = (2,1,1,5,5,)), "i261","i259","i262","i258","i260",)
	x276 = x279 * x280
	assert set(x276.index_ids) == set(["i259","i260","i261","i258","i262",])
	assert x276.release_array("i259","i260","i261","i258","i262",).shape == (1,5,2,5,1,)
	assert x276.rank == 5
	x268 = x275 * x276
	assert set(x268.index_ids) == set(["i259","i260","i262","i261","i258",])
	assert x268.release_array("i259","i260","i262","i261","i258",).shape == (1,5,1,2,5,)
	assert x268.rank == 5
	x265 = x267 * x268
	assert set(x265.index_ids) == set(["i258","i259",])
	assert x265.release_array("i258","i259",).shape == (5,1,)
	assert x265.rank == 2
	x285 = NXArray(normal(size = (3,4,3,2,1,)), "i277","i276","i275","i274","i273",)
	x286 = NXArray(normal(size = (1,3,)), "i255","i277",)
	x283 = x285 * x286
	assert set(x283.index_ids) == set(["i274","i273","i275","i276","i255",])
	assert x283.release_array("i274","i273","i275","i276","i255",).shape == (2,1,3,4,1,)
	assert x283.rank == 5
	x287 = NXArray(normal(size = (2,2,4,2,3,)), "i278","i279","i276","i274","i275",)
	x288 = NXArray(normal(size = (2,2,)), "i278","i279",)
	x284 = x287 * x288
	assert set(x284.index_ids) == set(["i276","i274","i275",])
	assert x284.release_array("i276","i274","i275",).shape == (4,2,3,)
	assert x284.rank == 3
	x281 = x283 * x284
	assert set(x281.index_ids) == set(["i255","i273",])
	assert x281.release_array("i255","i273",).shape == (1,1,)
	assert x281.rank == 2
	x291 = NXArray(normal(size = (2,2,)), "i280","i282",)
	x292 = NXArray(normal(size = (3,3,2,5,1,)), "i256","i281","i282","i257","i273",)
	x289 = x291 * x292
	assert set(x289.index_ids) == set(["i280","i257","i273","i256","i281",])
	assert x289.release_array("i280","i257","i273","i256","i281",).shape == (2,5,1,3,3,)
	assert x289.rank == 5
	x293 = NXArray(normal(size = (1,5,3,2,1,)), "i259","i258","i281","i280","i283",)
	x294 = NXArray(normal(size = (1,)), "i283",)
	x290 = x293 * x294
	assert set(x290.index_ids) == set(["i280","i281","i258","i259",])
	assert x290.release_array("i280","i281","i258","i259",).shape == (2,3,5,1,)
	assert x290.rank == 4
	x282 = x289 * x290
	assert set(x282.index_ids) == set(["i273","i257","i256","i258","i259",])
	assert x282.release_array("i273","i257","i256","i258","i259",).shape == (1,5,3,5,1,)
	assert x282.rank == 5
	x266 = x281 * x282
	assert set(x266.index_ids) == set(["i255","i258","i256","i257","i259",])
	assert x266.release_array("i255","i258","i256","i257","i259",).shape == (1,5,3,5,1,)
	assert x266.rank == 5
	x263 = x265 * x266
	assert set(x263.index_ids) == set(["i255","i257","i256",])
	assert x263.release_array("i255","i257","i256",).shape == (1,5,3,)
	assert x263.rank == 3
	x301 = NXArray(normal(size = (1,)), "i286",)
	x302 = NXArray(normal(size = (1,3,2,3,3,)), "i286","i285","i284","i256","i250",)
	x299 = x301 * x302
	assert set(x299.index_ids) == set(["i284","i250","i285","i256",])
	assert x299.release_array("i284","i250","i285","i256",).shape == (2,3,3,3,)
	assert x299.rank == 4
	x303 = NXArray(normal(size = (3,3,1,2,1,)), "i287","i288","i255","i284","i254",)
	x304 = NXArray(normal(size = (3,5,3,3,)), "i287","i257","i285","i288",)
	x300 = x303 * x304
	assert set(x300.index_ids) == set(["i255","i284","i254","i285","i257",])
	assert x300.release_array("i255","i284","i254","i285","i257",).shape == (1,2,1,3,5,)
	assert x300.rank == 5
	x297 = x299 * x300
	assert set(x297.index_ids) == set(["i250","i256","i254","i255","i257",])
	assert x297.release_array("i250","i256","i254","i255","i257",).shape == (3,3,1,1,5,)
	assert x297.rank == 5
	x307 = NXArray(normal(size = ()), )
	x308 = NXArray(normal(size = (3,5,3,1,2,)), "i292","i289","i291","i290","i293",)
	x305 = x307 * x308
	assert set(x305.index_ids) == set(["i292","i291","i289","i290","i293",])
	assert x305.release_array("i292","i291","i289","i290","i293",).shape == (3,3,5,1,2,)
	assert x305.rank == 5
	x309 = NXArray(normal(size = (1,5,3,2,1,)), "i294","i289","i292","i293","i290",)
	x310 = NXArray(normal(size = (1,3,)), "i294","i291",)
	x306 = x309 * x310
	assert set(x306.index_ids) == set(["i289","i292","i293","i290","i291",])
	assert x306.release_array("i289","i292","i293","i290","i291",).shape == (5,3,2,1,3,)
	assert x306.rank == 5
	x298 = x305 * x306
	assert set(x298.index_ids) == set([])
	assert x298.release_array().shape == ()
	assert x298.rank == 0
	x295 = x297 * x298
	assert set(x295.index_ids) == set(["i257","i254","i255","i256","i250",])
	assert x295.release_array("i257","i254","i255","i256","i250",).shape == (5,1,1,3,3,)
	assert x295.rank == 5
	x315 = NXArray(normal(size = (1,3,2,4,5,)), "i297","i301","i300","i303","i302",)
	x316 = NXArray(normal(size = (5,5,4,4,)), "i298","i302","i295","i303",)
	x313 = x315 * x316
	assert set(x313.index_ids) == set(["i300","i301","i297","i298","i295",])
	assert x313.release_array("i300","i301","i297","i298","i295",).shape == (2,3,1,5,4,)
	assert x313.rank == 5
	x317 = NXArray(normal(size = (4,3,1,)), "i304","i301","i305",)
	x318 = NXArray(normal(size = (2,4,1,1,2,)), "i300","i304","i296","i305","i299",)
	x314 = x317 * x318
	assert set(x314.index_ids) == set(["i301","i299","i296","i300",])
	assert x314.release_array("i301","i299","i296","i300",).shape == (3,2,1,2,)
	assert x314.rank == 4
	x311 = x313 * x314
	assert set(x311.index_ids) == set(["i297","i295","i298","i296","i299",])
	assert x311.release_array("i297","i295","i298","i296","i299",).shape == (1,4,5,1,2,)
	assert x311.rank == 5
	x321 = NXArray(normal(size = (2,4,2,3,4,)), "i308","i295","i306","i307","i309",)
	x322 = NXArray(normal(size = (2,4,3,)), "i308","i309","i307",)
	x319 = x321 * x322
	assert set(x319.index_ids) == set(["i306","i295",])
	assert x319.release_array("i306","i295",).shape == (2,4,)
	assert x319.rank == 2
	x323 = NXArray(normal(size = (4,2,4,1,1,)), "i310","i306","i311","i296","i297",)
	x324 = NXArray(normal(size = (4,5,2,4,)), "i310","i298","i299","i311",)
	x320 = x323 * x324
	assert set(x320.index_ids) == set(["i306","i296","i297","i299","i298",])
	assert x320.release_array("i306","i296","i297","i299","i298",).shape == (2,1,1,2,5,)
	assert x320.rank == 5
	x312 = x319 * x320
	assert set(x312.index_ids) == set(["i295","i296","i297","i299","i298",])
	assert x312.release_array("i295","i296","i297","i299","i298",).shape == (4,1,1,2,5,)
	assert x312.rank == 5
	x296 = x311 * x312
	assert set(x296.index_ids) == set([])
	assert x296.release_array().shape == ()
	assert x296.rank == 0
	x264 = x295 * x296
	assert set(x264.index_ids) == set(["i255","i254","i257","i256","i250",])
	assert x264.release_array("i255","i254","i257","i256","i250",).shape == (1,1,5,3,3,)
	assert x264.rank == 5
	x261 = x263 * x264
	assert set(x261.index_ids) == set(["i250","i254",])
	assert x261.release_array("i250","i254",).shape == (3,1,)
	assert x261.rank == 2
	x333 = NXArray(normal(size = (1,3,2,5,1,)), "i318","i317","i321","i319","i316",)
	x334 = NXArray(normal(size = (5,2,)), "i320","i321",)
	x331 = x333 * x334
	assert set(x331.index_ids) == set(["i316","i318","i319","i317","i320",])
	assert x331.release_array("i316","i318","i319","i317","i320",).shape == (1,1,5,3,5,)
	assert x331.rank == 5
	x335 = NXArray(normal(size = (5,5,5,1,3,)), "i319","i320","i322","i318","i323",)
	x336 = NXArray(normal(size = (5,3,)), "i322","i323",)
	x332 = x335 * x336
	assert set(x332.index_ids) == set(["i320","i318","i319",])
	assert x332.release_array("i320","i318","i319",).shape == (5,1,5,)
	assert x332.rank == 3
	x329 = x331 * x332
	assert set(x329.index_ids) == set(["i316","i317",])
	assert x329.release_array("i316","i317",).shape == (1,3,)
	assert x329.rank == 2
	x339 = NXArray(normal(size = (3,1,3,)), "i326","i325","i327",)
	x340 = NXArray(normal(size = (3,1,1,2,3,)), "i327","i316","i325","i324","i326",)
	x337 = x339 * x340
	assert set(x337.index_ids) == set(["i324","i316",])
	assert x337.release_array("i324","i316",).shape == (2,1,)
	assert x337.rank == 2
	x341 = NXArray(normal(size = (2,2,2,4,3,)), "i328","i312","i324","i315","i317",)
	x342 = NXArray(normal(size = (2,4,)), "i328","i314",)
	x338 = x341 * x342
	assert set(x338.index_ids) == set(["i324","i315","i317","i312","i314",])
	assert x338.release_array("i324","i315","i317","i312","i314",).shape == (2,4,3,2,4,)
	assert x338.rank == 5
	x330 = x337 * x338
	assert set(x330.index_ids) == set(["i316","i317","i312","i315","i314",])
	assert x330.release_array("i316","i317","i312","i315","i314",).shape == (1,3,2,4,4,)
	assert x330.rank == 5
	x327 = x329 * x330
	assert set(x327.index_ids) == set(["i315","i312","i314",])
	assert x327.release_array("i315","i312","i314",).shape == (4,2,4,)
	assert x327.rank == 3
	x347 = NXArray(normal(size = (4,2,4,)), "i334","i333","i332",)
	x348 = NXArray(normal(size = (2,1,4,4,4,)), "i333","i331","i334","i313","i314",)
	x345 = x347 * x348
	assert set(x345.index_ids) == set(["i332","i331","i314","i313",])
	assert x345.release_array("i332","i331","i314","i313",).shape == (4,1,4,4,)
	assert x345.rank == 4
	x349 = NXArray(normal(size = ()), )
	x350 = NXArray(normal(size = (1,1,2,1,4,)), "i254","i331","i330","i329","i332",)
	x346 = x349 * x350
	assert set(x346.index_ids) == set(["i330","i329","i331","i254","i332",])
	assert x346.release_array("i330","i329","i331","i254","i332",).shape == (2,1,1,1,4,)
	assert x346.rank == 5
	x343 = x345 * x346
	assert set(x343.index_ids) == set(["i314","i313","i329","i330","i254",])
	assert x343.release_array("i314","i313","i329","i330","i254",).shape == (4,4,1,2,1,)
	assert x343.rank == 5
	x353 = NXArray(normal(size = (4,5,)), "i315","i337",)
	x354 = NXArray(normal(size = (5,1,5,1,2,)), "i335","i329","i337","i336","i330",)
	x351 = x353 * x354
	assert set(x351.index_ids) == set(["i315","i336","i330","i335","i329",])
	assert x351.release_array("i315","i336","i330","i335","i329",).shape == (4,1,2,5,1,)
	assert x351.rank == 5
	x355 = NXArray(normal(size = (3,3,3,1,)), "i338","i253","i340","i339",)
	x356 = NXArray(normal(size = (3,1,1,3,5,)), "i340","i336","i339","i338","i335",)
	x352 = x355 * x356
	assert set(x352.index_ids) == set(["i253","i335","i336",])
	assert x352.release_array("i253","i335","i336",).shape == (3,5,1,)
	assert x352.rank == 3
	x344 = x351 * x352
	assert set(x344.index_ids) == set(["i329","i330","i315","i253",])
	assert x344.release_array("i329","i330","i315","i253",).shape == (1,2,4,3,)
	assert x344.rank == 4
	x328 = x343 * x344
	assert set(x328.index_ids) == set(["i254","i314","i313","i315","i253",])
	assert x328.release_array("i254","i314","i313","i315","i253",).shape == (1,4,4,4,3,)
	assert x328.rank == 5
	x325 = x327 * x328
	assert set(x325.index_ids) == set(["i312","i313","i253","i254",])
	assert x325.release_array("i312","i313","i253","i254",).shape == (2,4,3,1,)
	assert x325.rank == 4
	x363 = NXArray(normal(size = (5,1,1,2,1,)), "i343","i342","i344","i347","i346",)
	x364 = NXArray(normal(size = (1,2,1,2,)), "i346","i347","i252","i345",)
	x361 = x363 * x364
	assert set(x361.index_ids) == set(["i342","i343","i344","i252","i345",])
	assert x361.release_array("i342","i343","i344","i252","i345",).shape == (1,5,1,1,2,)
	assert x361.rank == 5
	x365 = NXArray(normal(size = (4,4,1,2,4,)), "i341","i349","i348","i345","i350",)
	x366 = NXArray(normal(size = (4,1,4,1,)), "i349","i344","i350","i348",)
	x362 = x365 * x366
	assert set(x362.index_ids) == set(["i341","i345","i344",])
	assert x362.release_array("i341","i345","i344",).shape == (4,2,1,)
	assert x362.rank == 3
	x359 = x361 * x362
	assert set(x359.index_ids) == set(["i342","i343","i252","i341",])
	assert x359.release_array("i342","i343","i252","i341",).shape == (1,5,1,4,)
	assert x359.rank == 4
	x369 = NXArray(normal(size = (3,5,4,4,1,)), "i353","i343","i352","i313","i342",)
	x370 = NXArray(normal(size = (3,4,5,3,)), "i353","i352","i251","i351",)
	x367 = x369 * x370
	assert set(x367.index_ids) == set(["i342","i343","i313","i251","i351",])
	assert x367.release_array("i342","i343","i313","i251","i351",).shape == (1,5,4,5,3,)
	assert x367.rank == 5
	x371 = NXArray(normal(size = (1,2,3,3,5,)), "i355","i312","i351","i354","i356",)
	x372 = NXArray(normal(size = (3,1,5,)), "i354","i355","i356",)
	x368 = x371 * x372
	assert set(x368.index_ids) == set(["i312","i351",])
	assert x368.release_array("i312","i351",).shape == (2,3,)
	assert x368.rank == 2
	x360 = x367 * x368
	assert set(x360.index_ids) == set(["i343","i342","i251","i313","i312",])
	assert x360.release_array("i343","i342","i251","i313","i312",).shape == (5,1,5,4,2,)
	assert x360.rank == 5
	x357 = x359 * x360
	assert set(x357.index_ids) == set(["i252","i341","i313","i251","i312",])
	assert x357.release_array("i252","i341","i313","i251","i312",).shape == (1,4,4,5,2,)
	assert x357.rank == 5
	x377 = NXArray(normal(size = (1,4,1,2,1,)), "i364","i362","i361","i360","i363",)
	x378 = NXArray(normal(size = (1,4,1,)), "i364","i362","i363",)
	x375 = x377 * x378
	assert set(x375.index_ids) == set(["i360","i361",])
	assert x375.release_array("i360","i361",).shape == (2,1,)
	assert x375.rank == 2
	x379 = NXArray(normal(size = ()), )
	x380 = NXArray(normal(size = (2,3,1,1,1,)), "i360","i359","i361","i358","i357",)
	x376 = x379 * x380
	assert set(x376.index_ids) == set(["i360","i359","i361","i358","i357",])
	assert x376.release_array("i360","i359","i361","i358","i357",).shape == (2,3,1,1,1,)
	assert x376.rank == 5
	x373 = x375 * x376
	assert set(x373.index_ids) == set(["i358","i357","i359",])
	assert x373.release_array("i358","i357","i359",).shape == (1,1,3,)
	assert x373.rank == 3
	x383 = NXArray(normal(size = (2,2,4,3,1,)), "i366","i365","i341","i359","i357",)
	x384 = NXArray(normal(size = (2,1,)), "i366","i358",)
	x381 = x383 * x384
	assert set(x381.index_ids) == set(["i365","i357","i359","i341","i358",])
	assert x381.release_array("i365","i357","i359","i341","i358",).shape == (2,1,3,4,1,)
	assert x381.rank == 5
	x385 = NXArray(normal(size = (4,5,2,)), "i369","i368","i367",)
	x386 = NXArray(normal(size = (4,5,2,5,2,)), "i369","i249","i365","i368","i367",)
	x382 = x385 * x386
	assert set(x382.index_ids) == set(["i249","i365",])
	assert x382.release_array("i249","i365",).shape == (5,2,)
	assert x382.rank == 2
	x374 = x381 * x382
	assert set(x374.index_ids) == set(["i359","i341","i357","i358","i249",])
	assert x374.release_array("i359","i341","i357","i358","i249",).shape == (3,4,1,1,5,)
	assert x374.rank == 5
	x358 = x373 * x374
	assert set(x358.index_ids) == set(["i341","i249",])
	assert x358.release_array("i341","i249",).shape == (4,5,)
	assert x358.rank == 2
	x326 = x357 * x358
	assert set(x326.index_ids) == set(["i252","i251","i313","i312","i249",])
	assert x326.release_array("i252","i251","i313","i312","i249",).shape == (1,5,4,2,5,)
	assert x326.rank == 5
	x262 = x325 * x326
	assert set(x262.index_ids) == set(["i253","i254","i249","i251","i252",])
	assert x262.release_array("i253","i254","i249","i251","i252",).shape == (3,1,5,5,1,)
	assert x262.rank == 5
	x259 = x261 * x262
	assert set(x259.index_ids) == set(["i250","i249","i251","i252","i253",])
	assert x259.release_array("i250","i249","i251","i252","i253",).shape == (3,5,5,1,3,)
	assert x259.rank == 5
	x397 = NXArray(normal(size = (3,4,2,2,2,)), "i379","i377","i380","i378","i381",)
	x398 = NXArray(normal(size = ()), )
	x395 = x397 * x398
	assert set(x395.index_ids) == set(["i379","i380","i378","i377","i381",])
	assert x395.release_array("i379","i380","i378","i377","i381",).shape == (3,2,2,4,2,)
	assert x395.rank == 5
	x399 = NXArray(normal(size = (2,1,1,)), "i380","i383","i382",)
	x400 = NXArray(normal(size = (1,2,3,2,1,)), "i382","i381","i379","i378","i383",)
	x396 = x399 * x400
	assert set(x396.index_ids) == set(["i380","i381","i379","i378",])
	assert x396.release_array("i380","i381","i379","i378",).shape == (2,2,3,2,)
	assert x396.rank == 4
	x393 = x395 * x396
	assert set(x393.index_ids) == set(["i377",])
	assert x393.release_array("i377",).shape == (4,)
	assert x393.rank == 1
	x403 = NXArray(normal(size = (5,4,1,2,5,)), "i385","i386","i388","i384","i387",)
	x404 = NXArray(normal(size = (4,5,2,5,1,)), "i386","i387","i384","i385","i388",)
	x401 = x403 * x404
	assert set(x401.index_ids) == set([])
	assert x401.release_array().shape == ()
	assert x401.rank == 0
	x405 = NXArray(normal(size = (4,1,)), "i377","i389",)
	x406 = NXArray(normal(size = (2,1,3,2,3,)), "i376","i389","i372","i375","i374",)
	x402 = x405 * x406
	assert set(x402.index_ids) == set(["i377","i374","i375","i372","i376",])
	assert x402.release_array("i377","i374","i375","i372","i376",).shape == (4,3,2,3,2,)
	assert x402.rank == 5
	x394 = x401 * x402
	assert set(x394.index_ids) == set(["i374","i375","i377","i376","i372",])
	assert x394.release_array("i374","i375","i377","i376","i372",).shape == (3,2,4,2,3,)
	assert x394.rank == 5
	x391 = x393 * x394
	assert set(x391.index_ids) == set(["i375","i374","i376","i372",])
	assert x391.release_array("i375","i374","i376","i372",).shape == (2,3,2,3,)
	assert x391.rank == 4
	x411 = NXArray(normal(size = (3,5,1,1,1,)), "i396","i395","i392","i394","i393",)
	x412 = NXArray(normal(size = (5,1,3,1,1,)), "i395","i394","i396","i393","i392",)
	x409 = x411 * x412
	assert set(x409.index_ids) == set([])
	assert x409.release_array().shape == ()
	assert x409.rank == 0
	x413 = NXArray(normal(size = (4,5,1,5,)), "i397","i390","i398","i373",)
	x414 = NXArray(normal(size = (5,2,1,1,4,)), "i370","i375","i398","i391","i397",)
	x410 = x413 * x414
	assert set(x410.index_ids) == set(["i373","i390","i391","i370","i375",])
	assert x410.release_array("i373","i390","i391","i370","i375",).shape == (5,5,1,5,2,)
	assert x410.rank == 5
	x407 = x409 * x410
	assert set(x407.index_ids) == set(["i390","i391","i373","i375","i370",])
	assert x407.release_array("i390","i391","i373","i375","i370",).shape == (5,1,5,2,5,)
	assert x407.rank == 5
	x417 = NXArray(normal(size = (5,3,5,1,)), "i401","i399","i402","i391",)
	x418 = NXArray(normal(size = (5,5,5,3,2,)), "i402","i390","i401","i400","i376",)
	x415 = x417 * x418
	assert set(x415.index_ids) == set(["i391","i399","i376","i400","i390",])
	assert x415.release_array("i391","i399","i376","i400","i390",).shape == (1,3,2,3,5,)
	assert x415.rank == 5
	x419 = NXArray(normal(size = (3,1,2,3,)), "i403","i404","i405","i399",)
	x420 = NXArray(normal(size = (3,1,2,3,2,)), "i403","i404","i405","i400","i371",)
	x416 = x419 * x420
	assert set(x416.index_ids) == set(["i399","i400","i371",])
	assert x416.release_array("i399","i400","i371",).shape == (3,3,2,)
	assert x416.rank == 3
	x408 = x415 * x416
	assert set(x408.index_ids) == set(["i390","i376","i391","i371",])
	assert x408.release_array("i390","i376","i391","i371",).shape == (5,2,1,2,)
	assert x408.rank == 4
	x392 = x407 * x408
	assert set(x392.index_ids) == set(["i375","i373","i370","i376","i371",])
	assert x392.release_array("i375","i373","i370","i376","i371",).shape == (2,5,5,2,2,)
	assert x392.rank == 5
	x389 = x391 * x392
	assert set(x389.index_ids) == set(["i372","i374","i370","i371","i373",])
	assert x389.release_array("i372","i374","i370","i371","i373",).shape == (3,3,5,2,5,)
	assert x389.rank == 5
	x427 = NXArray(normal(size = (5,5,2,1,3,)), "i407","i410","i406","i408","i409",)
	x428 = NXArray(normal(size = (2,1,3,5,5,)), "i406","i408","i409","i410","i407",)
	x425 = x427 * x428
	assert set(x425.index_ids) == set([])
	assert x425.release_array().shape == ()
	assert x425.rank == 0
	x429 = NXArray(normal(size = (1,5,2,3,3,)), "i411","i412","i371","i374","i372",)
	x430 = NXArray(normal(size = (5,1,5,5,)), "i370","i411","i373","i412",)
	x426 = x429 * x430
	assert set(x426.index_ids) == set(["i374","i371","i372","i373","i370",])
	assert x426.release_array("i374","i371","i372","i373","i370",).shape == (3,2,3,5,5,)
	assert x426.rank == 5
	x423 = x425 * x426
	assert set(x423.index_ids) == set(["i371","i370","i374","i373","i372",])
	assert x423.release_array("i371","i370","i374","i373","i372",).shape == (2,5,3,5,3,)
	assert x423.rank == 5
	x433 = NXArray(normal(size = (1,3,3,4,5,)), "i418","i416","i414","i413","i419",)
	x434 = NXArray(normal(size = (4,2,1,5,)), "i417","i415","i418","i419",)
	x431 = x433 * x434
	assert set(x431.index_ids) == set(["i413","i416","i414","i415","i417",])
	assert x431.release_array("i413","i416","i414","i415","i417",).shape == (4,3,3,2,4,)
	assert x431.rank == 5
	x435 = NXArray(normal(size = (4,3,5,3,3,)), "i417","i414","i420","i421","i416",)
	x436 = NXArray(normal(size = (4,5,2,3,)), "i413","i420","i415","i421",)
	x432 = x435 * x436
	assert set(x432.index_ids) == set(["i417","i416","i414","i415","i413",])
	assert x432.release_array("i417","i416","i414","i415","i413",).shape == (4,3,3,2,4,)
	assert x432.rank == 5
	x424 = x431 * x432
	assert set(x424.index_ids) == set([])
	assert x424.release_array().shape == ()
	assert x424.rank == 0
	x421 = x423 * x424
	assert set(x421.index_ids) == set(["i371","i370","i374","i373","i372",])
	assert x421.release_array("i371","i370","i374","i373","i372",).shape == (2,5,3,5,3,)
	assert x421.rank == 5
	x441 = NXArray(normal(size = (1,1,4,)), "i429","i430","i426",)
	x442 = NXArray(normal(size = (1,2,3,1,3,)), "i429","i422","i427","i430","i428",)
	x439 = x441 * x442
	assert set(x439.index_ids) == set(["i426","i428","i427","i422",])
	assert x439.release_array("i426","i428","i427","i422",).shape == (4,3,3,2,)
	assert x439.rank == 4
	x443 = NXArray(normal(size = (3,4,3,1,3,)), "i427","i431","i428","i423","i425",)
	x444 = NXArray(normal(size = (5,4,)), "i424","i431",)
	x440 = x443 * x444
	assert set(x440.index_ids) == set(["i425","i428","i427","i423","i424",])
	assert x440.release_array("i425","i428","i427","i423","i424",).shape == (3,3,3,1,5,)
	assert x440.rank == 5
	x437 = x439 * x440
	assert set(x437.index_ids) == set(["i426","i422","i423","i425","i424",])
	assert x437.release_array("i426","i422","i423","i425","i424",).shape == (4,2,1,3,5,)
	assert x437.rank == 5
	x447 = NXArray(normal(size = (4,4,3,5,1,)), "i432","i426","i425","i433","i423",)
	x448 = NXArray(normal(size = (5,5,2,4,)), "i424","i433","i422","i432",)
	x445 = x447 * x448
	assert set(x445.index_ids) == set(["i425","i423","i426","i424","i422",])
	assert x445.release_array("i425","i423","i426","i424","i422",).shape == (3,1,4,5,2,)
	assert x445.rank == 5
	x449 = NXArray(normal(size = (5,5,4,2,3,)), "i438","i436","i437","i434","i435",)
	x450 = NXArray(normal(size = (2,3,5,4,5,)), "i434","i435","i436","i437","i438",)
	x446 = x449 * x450
	assert set(x446.index_ids) == set([])
	assert x446.release_array().shape == ()
	assert x446.rank == 0
	x438 = x445 * x446
	assert set(x438.index_ids) == set(["i422","i423","i424","i426","i425",])
	assert x438.release_array("i422","i423","i424","i426","i425",).shape == (2,1,5,4,3,)
	assert x438.rank == 5
	x422 = x437 * x438
	assert set(x422.index_ids) == set([])
	assert x422.release_array().shape == ()
	assert x422.rank == 0
	x390 = x421 * x422
	assert set(x390.index_ids) == set(["i374","i373","i371","i370","i372",])
	assert x390.release_array("i374","i373","i371","i370","i372",).shape == (3,5,2,5,3,)
	assert x390.rank == 5
	x387 = x389 * x390
	assert set(x387.index_ids) == set([])
	assert x387.release_array().shape == ()
	assert x387.rank == 0
	x459 = NXArray(normal(size = (3,2,5,5,)), "i441","i439","i251","i442",)
	x460 = NXArray(normal(size = (3,3,5,5,3,)), "i441","i253","i442","i249","i440",)
	x457 = x459 * x460
	assert set(x457.index_ids) == set(["i439","i251","i253","i249","i440",])
	assert x457.release_array("i439","i251","i253","i249","i440",).shape == (2,5,3,5,3,)
	assert x457.rank == 5
	x461 = NXArray(normal(size = (4,1,4,5,2,)), "i443","i444","i445","i447","i446",)
	x462 = NXArray(normal(size = (4,1,2,5,4,)), "i443","i444","i446","i447","i445",)
	x458 = x461 * x462
	assert set(x458.index_ids) == set([])
	assert x458.release_array().shape == ()
	assert x458.rank == 0
	x455 = x457 * x458
	assert set(x455.index_ids) == set(["i253","i249","i440","i251","i439",])
	assert x455.release_array("i253","i249","i440","i251","i439",).shape == (3,5,3,5,2,)
	assert x455.rank == 5
	x465 = NXArray(normal(size = (3,3,5,1,1,)), "i440","i250","i451","i448","i449",)
	x466 = NXArray(normal(size = (5,4,)), "i451","i450",)
	x463 = x465 * x466
	assert set(x463.index_ids) == set(["i250","i449","i448","i440","i450",])
	assert x463.release_array("i250","i449","i448","i440","i450",).shape == (3,1,1,3,4,)
	assert x463.rank == 5
	x467 = NXArray(normal(size = (1,2,5,1,1,)), "i453","i452","i454","i448","i449",)
	x468 = NXArray(normal(size = (4,2,1,5,)), "i450","i452","i453","i454",)
	x464 = x467 * x468
	assert set(x464.index_ids) == set(["i448","i449","i450",])
	assert x464.release_array("i448","i449","i450",).shape == (1,1,4,)
	assert x464.rank == 3
	x456 = x463 * x464
	assert set(x456.index_ids) == set(["i250","i440",])
	assert x456.release_array("i250","i440",).shape == (3,3,)
	assert x456.rank == 2
	x453 = x455 * x456
	assert set(x453.index_ids) == set(["i249","i439","i253","i251","i250",])
	assert x453.release_array("i249","i439","i253","i251","i250",).shape == (5,2,3,5,3,)
	assert x453.rank == 5
	x473 = NXArray(normal(size = (5,2,2,1,3,)), "i460","i456","i439","i455","i459",)
	x474 = NXArray(normal(size = (1,5,)), "i457","i460",)
	x471 = x473 * x474
	assert set(x471.index_ids) == set(["i439","i455","i459","i456","i457",])
	assert x471.release_array("i439","i455","i459","i456","i457",).shape == (2,1,3,2,1,)
	assert x471.rank == 5
	x475 = NXArray(normal(size = (4,5,3,3,2,)), "i461","i463","i462","i459","i458",)
	x476 = NXArray(normal(size = (4,3,5,)), "i461","i462","i463",)
	x472 = x475 * x476
	assert set(x472.index_ids) == set(["i459","i458",])
	assert x472.release_array("i459","i458",).shape == (3,2,)
	assert x472.rank == 2
	x469 = x471 * x472
	assert set(x469.index_ids) == set(["i439","i455","i457","i456","i458",])
	assert x469.release_array("i439","i455","i457","i456","i458",).shape == (2,1,1,2,2,)
	assert x469.rank == 5
	x479 = NXArray(normal(size = (4,5,4,2,5,)), "i464","i468","i465","i466","i467",)
	x480 = NXArray(normal(size = (4,4,5,5,2,)), "i465","i464","i468","i467","i466",)
	x477 = x479 * x480
	assert set(x477.index_ids) == set([])
	assert x477.release_array().shape == ()
	assert x477.rank == 0
	x481 = NXArray(normal(size = ()), )
	x482 = NXArray(normal(size = (1,1,1,2,2,)), "i252","i455","i457","i456","i458",)
	x478 = x481 * x482
	assert set(x478.index_ids) == set(["i252","i455","i456","i457","i458",])
	assert x478.release_array("i252","i455","i456","i457","i458",).shape == (1,1,2,1,2,)
	assert x478.rank == 5
	x470 = x477 * x478
	assert set(x470.index_ids) == set(["i455","i456","i457","i458","i252",])
	assert x470.release_array("i455","i456","i457","i458","i252",).shape == (1,2,1,2,1,)
	assert x470.rank == 5
	x454 = x469 * x470
	assert set(x454.index_ids) == set(["i439","i252",])
	assert x454.release_array("i439","i252",).shape == (2,1,)
	assert x454.rank == 2
	x451 = x453 * x454
	assert set(x451.index_ids) == set(["i249","i253","i250","i251","i252",])
	assert x451.release_array("i249","i253","i250","i251","i252",).shape == (5,3,3,5,1,)
	assert x451.rank == 5
	x489 = NXArray(normal(size = (5,1,1,4,)), "i478","i479","i480","i477",)
	x490 = NXArray(normal(size = (1,2,5,1,2,)), "i479","i476","i478","i480","i475",)
	x487 = x489 * x490
	assert set(x487.index_ids) == set(["i477","i475","i476",])
	assert x487.release_array("i477","i475","i476",).shape == (4,2,2,)
	assert x487.rank == 3
	x491 = NXArray(normal(size = (2,5,2,4,1,)), "i476","i473","i475","i477","i474",)
	x492 = NXArray(normal(size = ()), )
	x488 = x491 * x492
	assert set(x488.index_ids) == set(["i476","i473","i474","i475","i477",])
	assert x488.release_array("i476","i473","i474","i475","i477",).shape == (2,5,1,2,4,)
	assert x488.rank == 5
	x485 = x487 * x488
	assert set(x485.index_ids) == set(["i474","i473",])
	assert x485.release_array("i474","i473",).shape == (1,5,)
	assert x485.rank == 2
	x495 = NXArray(normal(size = (5,5,1,4,5,)), "i485","i481","i483","i482","i484",)
	x496 = NXArray(normal(size = (5,5,5,1,4,)), "i481","i484","i485","i483","i482",)
	x493 = x495 * x496
	assert set(x493.index_ids) == set([])
	assert x493.release_array().shape == ()
	assert x493.rank == 0
	x497 = NXArray(normal(size = (5,4,5,1,4,)), "i471","i472","i469","i474","i486",)
	x498 = NXArray(normal(size = (4,4,)), "i486","i470",)
	x494 = x497 * x498
	assert set(x494.index_ids) == set(["i469","i472","i474","i471","i470",])
	assert x494.release_array("i469","i472","i474","i471","i470",).shape == (5,4,1,5,4,)
	assert x494.rank == 5
	x486 = x493 * x494
	assert set(x486.index_ids) == set(["i471","i470","i474","i469","i472",])
	assert x486.release_array("i471","i470","i474","i469","i472",).shape == (5,4,1,5,4,)
	assert x486.rank == 5
	x483 = x485 * x486
	assert set(x483.index_ids) == set(["i473","i471","i469","i472","i470",])
	assert x483.release_array("i473","i471","i469","i472","i470",).shape == (5,5,5,4,4,)
	assert x483.rank == 5
	x503 = NXArray(normal(size = (1,4,3,1,3,)), "i490","i491","i487","i488","i489",)
	x504 = NXArray(normal(size = (3,3,4,1,1,)), "i487","i489","i491","i490","i488",)
	x501 = x503 * x504
	assert set(x501.index_ids) == set([])
	assert x501.release_array().shape == ()
	assert x501.rank == 0
	x505 = NXArray(normal(size = (3,5,4,4,5,)), "i492","i473","i493","i472","i471",)
	x506 = NXArray(normal(size = (3,5,4,4,)), "i492","i469","i470","i493",)
	x502 = x505 * x506
	assert set(x502.index_ids) == set(["i473","i471","i472","i470","i469",])
	assert x502.release_array("i473","i471","i472","i470","i469",).shape == (5,5,4,4,5,)
	assert x502.rank == 5
	x499 = x501 * x502
	assert set(x499.index_ids) == set(["i469","i473","i472","i471","i470",])
	assert x499.release_array("i469","i473","i472","i471","i470",).shape == (5,5,4,5,4,)
	assert x499.rank == 5
	x509 = NXArray(normal(size = (4,4,4,3,5,)), "i498","i495","i499","i500","i494",)
	x510 = NXArray(normal(size = (2,4,3,4,)), "i496","i499","i500","i497",)
	x507 = x509 * x510
	assert set(x507.index_ids) == set(["i495","i494","i498","i497","i496",])
	assert x507.release_array("i495","i494","i498","i497","i496",).shape == (4,5,4,4,2,)
	assert x507.rank == 5
	x511 = NXArray(normal(size = (4,1,2,1,5,)), "i495","i501","i496","i502","i494",)
	x512 = NXArray(normal(size = (4,1,4,1,)), "i497","i502","i498","i501",)
	x508 = x511 * x512
	assert set(x508.index_ids) == set(["i494","i495","i496","i497","i498",])
	assert x508.release_array("i494","i495","i496","i497","i498",).shape == (5,4,2,4,4,)
	assert x508.rank == 5
	x500 = x507 * x508
	assert set(x500.index_ids) == set([])
	assert x500.release_array().shape == ()
	assert x500.rank == 0
	x484 = x499 * x500
	assert set(x484.index_ids) == set(["i471","i470","i472","i469","i473",])
	assert x484.release_array("i471","i470","i472","i469","i473",).shape == (5,4,4,5,5,)
	assert x484.rank == 5
	x452 = x483 * x484
	assert set(x452.index_ids) == set([])
	assert x452.release_array().shape == ()
	assert x452.rank == 0
	x388 = x451 * x452
	assert set(x388.index_ids) == set(["i250","i249","i251","i253","i252",])
	assert x388.release_array("i250","i249","i251","i253","i252",).shape == (3,5,5,3,1,)
	assert x388.rank == 5
	x260 = x387 * x388
	assert set(x260.index_ids) == set(["i251","i249","i253","i250","i252",])
	assert x260.release_array("i251","i249","i253","i250","i252",).shape == (5,5,3,3,1,)
	assert x260.rank == 5
	x4 = x259 * x260
	assert set(x4.index_ids) == set([])
	assert x4.release_array().shape == ()
	assert x4.rank == 0
	x1 = x3 * x4
	assert set(x1.index_ids) == set(["i1","i6","i5","i0","i2",])
	assert x1.release_array("i1","i6","i5","i0","i2",).shape == (2,4,3,3,3,)
	assert x1.rank == 5
	x527 = NXArray(normal(size = (5,5,2,1,3,)), "i512","i514","i513","i510","i508",)
	x528 = NXArray(normal(size = (5,1,)), "i514","i511",)
	x525 = x527 * x528
	assert set(x525.index_ids) == set(["i513","i508","i512","i510","i511",])
	assert x525.release_array("i513","i508","i512","i510","i511",).shape == (2,3,5,1,1,)
	assert x525.rank == 5
	x529 = NXArray(normal(size = (2,2,5,5,3,)), "i516","i515","i519","i517","i518",)
	x530 = NXArray(normal(size = (5,2,2,5,3,)), "i519","i515","i516","i517","i518",)
	x526 = x529 * x530
	assert set(x526.index_ids) == set([])
	assert x526.release_array().shape == ()
	assert x526.rank == 0
	x523 = x525 * x526
	assert set(x523.index_ids) == set(["i511","i510","i508","i513","i512",])
	assert x523.release_array("i511","i510","i508","i513","i512",).shape == (1,1,3,2,5,)
	assert x523.rank == 5
	x533 = NXArray(normal(size = (4,1,1,3,2,)), "i523","i520","i511","i521","i522",)
	x534 = NXArray(normal(size = (2,4,2,)), "i522","i523","i513",)
	x531 = x533 * x534
	assert set(x531.index_ids) == set(["i521","i520","i511","i513",])
	assert x531.release_array("i521","i520","i511","i513",).shape == (3,1,1,2,)
	assert x531.rank == 4
	x535 = NXArray(normal(size = (5,1,5,1,3,)), "i512","i520","i524","i510","i521",)
	x536 = NXArray(normal(size = (5,5,)), "i524","i509",)
	x532 = x535 * x536
	assert set(x532.index_ids) == set(["i520","i510","i521","i512","i509",])
	assert x532.release_array("i520","i510","i521","i512","i509",).shape == (1,1,3,5,5,)
	assert x532.rank == 5
	x524 = x531 * x532
	assert set(x524.index_ids) == set(["i513","i511","i512","i510","i509",])
	assert x524.release_array("i513","i511","i512","i510","i509",).shape == (2,1,5,1,5,)
	assert x524.rank == 5
	x521 = x523 * x524
	assert set(x521.index_ids) == set(["i508","i509",])
	assert x521.release_array("i508","i509",).shape == (3,5,)
	assert x521.rank == 2
	x541 = NXArray(normal(size = (4,2,2,5,4,)), "i531","i527","i529","i530","i528",)
	x542 = NXArray(normal(size = (3,2,5,4,3,)), "i505","i529","i530","i531","i526",)
	x539 = x541 * x542
	assert set(x539.index_ids) == set(["i527","i528","i505","i526",])
	assert x539.release_array("i527","i528","i505","i526",).shape == (2,4,3,3,)
	assert x539.rank == 4
	x543 = NXArray(normal(size = (4,4,2,1,5,)), "i506","i528","i527","i507","i525",)
	x544 = NXArray(normal(size = ()), )
	x540 = x543 * x544
	assert set(x540.index_ids) == set(["i527","i528","i506","i507","i525",])
	assert x540.release_array("i527","i528","i506","i507","i525",).shape == (2,4,4,1,5,)
	assert x540.rank == 5
	x537 = x539 * x540
	assert set(x537.index_ids) == set(["i526","i505","i525","i507","i506",])
	assert x537.release_array("i526","i505","i525","i507","i506",).shape == (3,3,5,1,4,)
	assert x537.rank == 5
	x547 = NXArray(normal(size = (4,4,5,1,4,)), "i533","i534","i535","i536","i532",)
	x548 = NXArray(normal(size = (1,4,4,5,)), "i536","i533","i534","i535",)
	x545 = x547 * x548
	assert set(x545.index_ids) == set(["i532",])
	assert x545.release_array("i532",).shape == (4,)
	assert x545.rank == 1
	x549 = NXArray(normal(size = (3,4,4,5,5,)), "i526","i537","i532","i525","i509",)
	x550 = NXArray(normal(size = (4,4,)), "i504","i537",)
	x546 = x549 * x550
	assert set(x546.index_ids) == set(["i509","i526","i532","i525","i504",])
	assert x546.release_array("i509","i526","i532","i525","i504",).shape == (5,3,4,5,4,)
	assert x546.rank == 5
	x538 = x545 * x546
	assert set(x538.index_ids) == set(["i525","i526","i509","i504",])
	assert x538.release_array("i525","i526","i509","i504",).shape == (5,3,5,4,)
	assert x538.rank == 4
	x522 = x537 * x538
	assert set(x522.index_ids) == set(["i507","i505","i506","i509","i504",])
	assert x522.release_array("i507","i505","i506","i509","i504",).shape == (1,3,4,5,4,)
	assert x522.rank == 5
	x519 = x521 * x522
	assert set(x519.index_ids) == set(["i508","i504","i506","i507","i505",])
	assert x519.release_array("i508","i504","i506","i507","i505",).shape == (3,4,4,1,3,)
	assert x519.rank == 5
	x557 = NXArray(normal(size = (5,2,4,3,)), "i547","i545","i548","i549",)
	x558 = NXArray(normal(size = (5,4,3,4,2,)), "i547","i548","i549","i546","i544",)
	x555 = x557 * x558
	assert set(x555.index_ids) == set(["i545","i544","i546",])
	assert x555.release_array("i545","i544","i546",).shape == (2,2,4,)
	assert x555.rank == 3
	x559 = NXArray(normal(size = (4,2,5,2,)), "i551","i543","i550","i545",)
	x560 = NXArray(normal(size = (5,1,4,4,2,)), "i550","i539","i551","i546","i544",)
	x556 = x559 * x560
	assert set(x556.index_ids) == set(["i543","i545","i544","i546","i539",])
	assert x556.release_array("i543","i545","i544","i546","i539",).shape == (2,2,2,4,1,)
	assert x556.rank == 5
	x553 = x555 * x556
	assert set(x553.index_ids) == set(["i539","i543",])
	assert x553.release_array("i539","i543",).shape == (1,2,)
	assert x553.rank == 2
	x563 = NXArray(normal(size = (1,3,5,2,2,)), "i553","i552","i554","i555","i543",)
	x564 = NXArray(normal(size = (2,5,3,)), "i555","i554","i538",)
	x561 = x563 * x564
	assert set(x561.index_ids) == set(["i543","i553","i552","i538",])
	assert x561.release_array("i543","i553","i552","i538",).shape == (2,1,3,3,)
	assert x561.rank == 4
	x565 = NXArray(normal(size = (3,3,)), "i556","i552",)
	x566 = NXArray(normal(size = (4,2,3,5,1,)), "i541","i540","i556","i542","i553",)
	x562 = x565 * x566
	assert set(x562.index_ids) == set(["i552","i553","i540","i542","i541",])
	assert x562.release_array("i552","i553","i540","i542","i541",).shape == (3,1,2,5,4,)
	assert x562.rank == 5
	x554 = x561 * x562
	assert set(x554.index_ids) == set(["i543","i538","i540","i542","i541",])
	assert x554.release_array("i543","i538","i540","i542","i541",).shape == (2,3,2,5,4,)
	assert x554.rank == 5
	x551 = x553 * x554
	assert set(x551.index_ids) == set(["i539","i538","i540","i541","i542",])
	assert x551.release_array("i539","i538","i540","i541","i542",).shape == (1,3,2,4,5,)
	assert x551.rank == 5
	x571 = NXArray(normal(size = (4,5,1,5,)), "i561","i562","i557","i542",)
	x572 = NXArray(normal(size = (4,2,5,4,4,)), "i561","i560","i562","i559","i558",)
	x569 = x571 * x572
	assert set(x569.index_ids) == set(["i542","i557","i560","i559","i558",])
	assert x569.release_array("i542","i557","i560","i559","i558",).shape == (5,1,2,4,4,)
	assert x569.rank == 5
	x573 = NXArray(normal(size = (4,4,2,2,4,)), "i559","i563","i565","i564","i558",)
	x574 = NXArray(normal(size = (2,2,4,2,)), "i560","i565","i563","i564",)
	x570 = x573 * x574
	assert set(x570.index_ids) == set(["i559","i558","i560",])
	assert x570.release_array("i559","i558","i560",).shape == (4,4,2,)
	assert x570.rank == 3
	x567 = x569 * x570
	assert set(x567.index_ids) == set(["i557","i542",])
	assert x567.release_array("i557","i542",).shape == (1,5,)
	assert x567.rank == 2
	x577 = NXArray(normal(size = (5,3,)), "i567","i538",)
	x578 = NXArray(normal(size = (1,5,2,5,4,)), "i539","i567","i540","i566","i541",)
	x575 = x577 * x578
	assert set(x575.index_ids) == set(["i538","i540","i541","i539","i566",])
	assert x575.release_array("i538","i540","i541","i539","i566",).shape == (3,2,4,1,5,)
	assert x575.rank == 5
	x579 = NXArray(normal(size = (1,1,5,2,1,)), "i568","i557","i566","i569","i570",)
	x580 = NXArray(normal(size = (1,2,1,)), "i570","i569","i568",)
	x576 = x579 * x580
	assert set(x576.index_ids) == set(["i557","i566",])
	assert x576.release_array("i557","i566",).shape == (1,5,)
	assert x576.rank == 2
	x568 = x575 * x576
	assert set(x568.index_ids) == set(["i541","i540","i539","i538","i557",])
	assert x568.release_array("i541","i540","i539","i538","i557",).shape == (4,2,1,3,1,)
	assert x568.rank == 5
	x552 = x567 * x568
	assert set(x552.index_ids) == set(["i542","i540","i539","i541","i538",])
	assert x552.release_array("i542","i540","i539","i541","i538",).shape == (5,2,1,4,3,)
	assert x552.rank == 5
	x520 = x551 * x552
	assert set(x520.index_ids) == set([])
	assert x520.release_array().shape == ()
	assert x520.rank == 0
	x517 = x519 * x520
	assert set(x517.index_ids) == set(["i506","i504","i508","i507","i505",])
	assert x517.release_array("i506","i504","i508","i507","i505",).shape == (4,4,3,1,3,)
	assert x517.rank == 5
	x589 = NXArray(normal(size = (4,1,3,1,5,)), "i579","i581","i578","i580","i582",)
	x590 = NXArray(normal(size = (1,1,3,4,5,)), "i581","i580","i578","i579","i582",)
	x587 = x589 * x590
	assert set(x587.index_ids) == set([])
	assert x587.release_array().shape == ()
	assert x587.rank == 0
	x591 = NXArray(normal(size = (3,5,1,5,)), "i573","i583","i577","i584",)
	x592 = NXArray(normal(size = (5,5,5,5,3,)), "i575","i574","i584","i583","i576",)
	x588 = x591 * x592
	assert set(x588.index_ids) == set(["i573","i577","i575","i574","i576",])
	assert x588.release_array("i573","i577","i575","i574","i576",).shape == (3,1,5,5,3,)
	assert x588.rank == 5
	x585 = x587 * x588
	assert set(x585.index_ids) == set(["i573","i574","i576","i577","i575",])
	assert x585.release_array("i573","i574","i576","i577","i575",).shape == (3,5,3,1,5,)
	assert x585.rank == 5
	x595 = NXArray(normal(size = (5,5,)), "i588","i585",)
	x596 = NXArray(normal(size = (5,5,3,1,2,)), "i586","i588","i576","i577","i587",)
	x593 = x595 * x596
	assert set(x593.index_ids) == set(["i585","i587","i577","i576","i586",])
	assert x593.release_array("i585","i587","i577","i576","i586",).shape == (5,2,1,3,5,)
	assert x593.rank == 5
	x597 = NXArray(normal(size = (1,5,1,1,)), "i590","i585","i591","i589",)
	x598 = NXArray(normal(size = (1,1,5,1,2,)), "i591","i590","i586","i589","i587",)
	x594 = x597 * x598
	assert set(x594.index_ids) == set(["i585","i587","i586",])
	assert x594.release_array("i585","i587","i586",).shape == (5,2,5,)
	assert x594.rank == 3
	x586 = x593 * x594
	assert set(x586.index_ids) == set(["i576","i577",])
	assert x586.release_array("i576","i577",).shape == (3,1,)
	assert x586.rank == 2
	x583 = x585 * x586
	assert set(x583.index_ids) == set(["i575","i574","i573",])
	assert x583.release_array("i575","i574","i573",).shape == (5,5,3,)
	assert x583.rank == 3
	x603 = NXArray(normal(size = (3,3,1,5,3,)), "i594","i595","i593","i592","i596",)
	x604 = NXArray(normal(size = (3,3,3,)), "i595","i594","i596",)
	x601 = x603 * x604
	assert set(x601.index_ids) == set(["i592","i593",])
	assert x601.release_array("i592","i593",).shape == (5,1,)
	assert x601.rank == 2
	x605 = NXArray(normal(size = ()), )
	x606 = NXArray(normal(size = (3,5,5,1,3,)), "i571","i574","i575","i593","i573",)
	x602 = x605 * x606
	assert set(x602.index_ids) == set(["i593","i571","i575","i573","i574",])
	assert x602.release_array("i593","i571","i575","i573","i574",).shape == (1,3,5,3,5,)
	assert x602.rank == 5
	x599 = x601 * x602
	assert set(x599.index_ids) == set(["i592","i571","i574","i573","i575",])
	assert x599.release_array("i592","i571","i574","i573","i575",).shape == (5,3,5,3,5,)
	assert x599.rank == 5
	x609 = NXArray(normal(size = (4,5,4,1,)), "i600","i592","i572","i601",)
	x610 = NXArray(normal(size = (3,5,4,1,2,)), "i599","i597","i600","i601","i598",)
	x607 = x609 * x610
	assert set(x607.index_ids) == set(["i572","i592","i599","i598","i597",])
	assert x607.release_array("i572","i592","i599","i598","i597",).shape == (4,5,3,2,5,)
	assert x607.rank == 5
	x611 = NXArray(normal(size = (4,3,4,5,2,)), "i602","i599","i603","i597","i598",)
	x612 = NXArray(normal(size = (4,4,)), "i603","i602",)
	x608 = x611 * x612
	assert set(x608.index_ids) == set(["i597","i599","i598",])
	assert x608.release_array("i597","i599","i598",).shape == (5,3,2,)
	assert x608.rank == 3
	x600 = x607 * x608
	assert set(x600.index_ids) == set(["i592","i572",])
	assert x600.release_array("i592","i572",).shape == (5,4,)
	assert x600.rank == 2
	x584 = x599 * x600
	assert set(x584.index_ids) == set(["i571","i573","i574","i575","i572",])
	assert x584.release_array("i571","i573","i574","i575","i572",).shape == (3,3,5,5,4,)
	assert x584.rank == 5
	x581 = x583 * x584
	assert set(x581.index_ids) == set(["i571","i572",])
	assert x581.release_array("i571","i572",).shape == (3,4,)
	assert x581.rank == 2
	x619 = NXArray(normal(size = (4,3,5,1,1,)), "i609","i604","i610","i612","i611",)
	x620 = NXArray(normal(size = (3,1,1,)), "i606","i611","i612",)
	x617 = x619 * x620
	assert set(x617.index_ids) == set(["i610","i604","i609","i606",])
	assert x617.release_array("i610","i604","i609","i606",).shape == (5,3,4,3,)
	assert x617.rank == 4
	x621 = NXArray(normal(size = ()), )
	x622 = NXArray(normal(size = (2,5,5,2,4,)), "i607","i503","i610","i608","i609",)
	x618 = x621 * x622
	assert set(x618.index_ids) == set(["i608","i607","i609","i610","i503",])
	assert x618.release_array("i608","i607","i609","i610","i503",).shape == (2,2,4,5,5,)
	assert x618.rank == 5
	x615 = x617 * x618
	assert set(x615.index_ids) == set(["i606","i604","i608","i607","i503",])
	assert x615.release_array("i606","i604","i608","i607","i503",).shape == (3,3,2,2,5,)
	assert x615.rank == 5
	x625 = NXArray(normal(size = (2,1,2,2,4,)), "i613","i614","i617","i615","i616",)
	x626 = NXArray(normal(size = (1,2,4,2,2,)), "i614","i613","i616","i617","i615",)
	x623 = x625 * x626
	assert set(x623.index_ids) == set([])
	assert x623.release_array().shape == ()
	assert x623.rank == 0
	x627 = NXArray(normal(size = (2,4,)), "i618","i605",)
	x628 = NXArray(normal(size = (2,2,3,2,3,)), "i608","i607","i606","i618","i571",)
	x624 = x627 * x628
	assert set(x624.index_ids) == set(["i605","i606","i607","i608","i571",])
	assert x624.release_array("i605","i606","i607","i608","i571",).shape == (4,3,2,2,3,)
	assert x624.rank == 5
	x616 = x623 * x624
	assert set(x616.index_ids) == set(["i606","i608","i607","i571","i605",])
	assert x616.release_array("i606","i608","i607","i571","i605",).shape == (3,2,2,3,4,)
	assert x616.rank == 5
	x613 = x615 * x616
	assert set(x613.index_ids) == set(["i604","i503","i605","i571",])
	assert x613.release_array("i604","i503","i605","i571",).shape == (3,5,4,3,)
	assert x613.rank == 4
	x633 = NXArray(normal(size = (4,3,2,2,)), "i622","i621","i624","i625",)
	x634 = NXArray(normal(size = (2,3,2,4,4,)), "i624","i620","i625","i623","i619",)
	x631 = x633 * x634
	assert set(x631.index_ids) == set(["i622","i621","i619","i620","i623",])
	assert x631.release_array("i622","i621","i619","i620","i623",).shape == (4,3,4,3,4,)
	assert x631.rank == 5
	x635 = NXArray(normal(size = (4,4,5,4,3,)), "i622","i623","i627","i619","i626",)
	x636 = NXArray(normal(size = (3,3,3,5,)), "i626","i621","i620","i627",)
	x632 = x635 * x636
	assert set(x632.index_ids) == set(["i623","i619","i622","i620","i621",])
	assert x632.release_array("i623","i619","i622","i620","i621",).shape == (4,4,4,3,3,)
	assert x632.rank == 5
	x629 = x631 * x632
	assert set(x629.index_ids) == set([])
	assert x629.release_array().shape == ()
	assert x629.rank == 0
	x639 = NXArray(normal(size = ()), )
	x640 = NXArray(normal(size = (4,3,3,1,4,)), "i572","i604","i508","i507","i605",)
	x637 = x639 * x640
	assert set(x637.index_ids) == set(["i572","i508","i604","i507","i605",])
	assert x637.release_array("i572","i508","i604","i507","i605",).shape == (4,3,3,1,4,)
	assert x637.rank == 5
	x641 = NXArray(normal(size = (3,3,4,4,2,)), "i628","i629","i631","i632","i630",)
	x642 = NXArray(normal(size = (3,4,3,4,2,)), "i628","i631","i629","i632","i630",)
	x638 = x641 * x642
	assert set(x638.index_ids) == set([])
	assert x638.release_array().shape == ()
	assert x638.rank == 0
	x630 = x637 * x638
	assert set(x630.index_ids) == set(["i604","i507","i508","i605","i572",])
	assert x630.release_array("i604","i507","i508","i605","i572",).shape == (3,1,3,4,4,)
	assert x630.rank == 5
	x614 = x629 * x630
	assert set(x614.index_ids) == set(["i604","i507","i508","i605","i572",])
	assert x614.release_array("i604","i507","i508","i605","i572",).shape == (3,1,3,4,4,)
	assert x614.rank == 5
	x582 = x613 * x614
	assert set(x582.index_ids) == set(["i503","i571","i507","i508","i572",])
	assert x582.release_array("i503","i571","i507","i508","i572",).shape == (5,3,1,3,4,)
	assert x582.rank == 5
	x518 = x581 * x582
	assert set(x518.index_ids) == set(["i503","i508","i507",])
	assert x518.release_array("i503","i508","i507",).shape == (5,3,1,)
	assert x518.rank == 3
	x515 = x517 * x518
	assert set(x515.index_ids) == set(["i504","i506","i505","i503",])
	assert x515.release_array("i504","i506","i505","i503",).shape == (4,4,3,5,)
	assert x515.rank == 4
	x653 = NXArray(normal(size = (3,1,4,1,)), "i646","i644","i645","i641",)
	x654 = NXArray(normal(size = (4,1,4,3,4,)), "i643","i644","i642","i646","i645",)
	x651 = x653 * x654
	assert set(x651.index_ids) == set(["i641","i642","i643",])
	assert x651.release_array("i641","i642","i643",).shape == (1,4,4,)
	assert x651.rank == 3
	x655 = NXArray(normal(size = ()), )
	x656 = NXArray(normal(size = (2,4,1,4,2,)), "i637","i643","i641","i642","i640",)
	x652 = x655 * x656
	assert set(x652.index_ids) == set(["i643","i642","i637","i641","i640",])
	assert x652.release_array("i643","i642","i637","i641","i640",).shape == (4,4,2,1,2,)
	assert x652.rank == 5
	x649 = x651 * x652
	assert set(x649.index_ids) == set(["i640","i637",])
	assert x649.release_array("i640","i637",).shape == (2,2,)
	assert x649.rank == 2
	x659 = NXArray(normal(size = (3,3,1,4,)), "i636","i647","i649","i648",)
	x660 = NXArray(normal(size = (5,2,4,1,4,)), "i635","i640","i648","i649","i639",)
	x657 = x659 * x660
	assert set(x657.index_ids) == set(["i636","i647","i635","i639","i640",])
	assert x657.release_array("i636","i647","i635","i639","i640",).shape == (3,3,5,4,2,)
	assert x657.rank == 5
	x661 = NXArray(normal(size = (3,4,4,)), "i652","i650","i651",)
	x662 = NXArray(normal(size = (4,3,4,3,2,)), "i651","i647","i650","i652","i638",)
	x658 = x661 * x662
	assert set(x658.index_ids) == set(["i638","i647",])
	assert x658.release_array("i638","i647",).shape == (2,3,)
	assert x658.rank == 2
	x650 = x657 * x658
	assert set(x650.index_ids) == set(["i636","i639","i640","i635","i638",])
	assert x650.release_array("i636","i639","i640","i635","i638",).shape == (3,4,2,5,2,)
	assert x650.rank == 5
	x647 = x649 * x650
	assert set(x647.index_ids) == set(["i637","i635","i638","i639","i636",])
	assert x647.release_array("i637","i635","i638","i639","i636",).shape == (2,5,2,4,3,)
	assert x647.rank == 5
	x667 = NXArray(normal(size = (1,1,2,)), "i659","i658","i657",)
	x668 = NXArray(normal(size = (1,1,2,1,3,)), "i656","i658","i657","i659","i655",)
	x665 = x667 * x668
	assert set(x665.index_ids) == set(["i655","i656",])
	assert x665.release_array("i655","i656",).shape == (3,1,)
	assert x665.rank == 2
	x669 = NXArray(normal(size = (4,1,2,5,2,)), "i660","i656","i638","i661","i654",)
	x670 = NXArray(normal(size = (5,3,4,5,)), "i653","i655","i660","i661",)
	x666 = x669 * x670
	assert set(x666.index_ids) == set(["i638","i654","i656","i653","i655",])
	assert x666.release_array("i638","i654","i656","i653","i655",).shape == (2,2,1,5,3,)
	assert x666.rank == 5
	x663 = x665 * x666
	assert set(x663.index_ids) == set(["i654","i638","i653",])
	assert x663.release_array("i654","i638","i653",).shape == (2,2,5,)
	assert x663.rank == 3
	x673 = NXArray(normal(size = (5,5,4,1,1,)), "i653","i664","i639","i662","i665",)
	x674 = NXArray(normal(size = (4,1,5,)), "i663","i665","i664",)
	x671 = x673 * x674
	assert set(x671.index_ids) == set(["i653","i662","i639","i663",])
	assert x671.release_array("i653","i662","i639","i663",).shape == (5,1,4,4,)
	assert x671.rank == 4
	x675 = NXArray(normal(size = (1,4,)), "i666","i663",)
	x676 = NXArray(normal(size = (5,1,2,1,2,)), "i633","i666","i634","i662","i654",)
	x672 = x675 * x676
	assert set(x672.index_ids) == set(["i663","i662","i633","i654","i634",])
	assert x672.release_array("i663","i662","i633","i654","i634",).shape == (4,1,5,2,2,)
	assert x672.rank == 5
	x664 = x671 * x672
	assert set(x664.index_ids) == set(["i653","i639","i654","i634","i633",])
	assert x664.release_array("i653","i639","i654","i634","i633",).shape == (5,4,2,2,5,)
	assert x664.rank == 5
	x648 = x663 * x664
	assert set(x648.index_ids) == set(["i638","i633","i639","i634",])
	assert x648.release_array("i638","i633","i639","i634",).shape == (2,5,4,2,)
	assert x648.rank == 4
	x645 = x647 * x648
	assert set(x645.index_ids) == set(["i637","i636","i635","i634","i633",])
	assert x645.release_array("i637","i636","i635","i634","i633",).shape == (2,3,5,2,5,)
	assert x645.rank == 5
	x683 = NXArray(normal(size = (1,2,3,3,1,)), "i670","i674","i672","i673","i671",)
	x684 = NXArray(normal(size = (2,)), "i674",)
	x681 = x683 * x684
	assert set(x681.index_ids) == set(["i672","i673","i671","i670",])
	assert x681.release_array("i672","i673","i671","i670",).shape == (3,3,1,1,)
	assert x681.rank == 4
	x685 = NXArray(normal(size = (5,1,3,2,3,)), "i668","i671","i673","i669","i672",)
	x686 = NXArray(normal(size = ()), )
	x682 = x685 * x686
	assert set(x682.index_ids) == set(["i673","i671","i669","i668","i672",])
	assert x682.release_array("i673","i671","i669","i668","i672",).shape == (3,1,2,5,3,)
	assert x682.rank == 5
	x679 = x681 * x682
	assert set(x679.index_ids) == set(["i670","i668","i669",])
	assert x679.release_array("i670","i668","i669",).shape == (1,5,2,)
	assert x679.rank == 3
	x689 = NXArray(normal(size = (2,2,)), "i669","i675",)
	x690 = NXArray(normal(size = (5,5,2,3,1,)), "i668","i667","i675","i636","i670",)
	x687 = x689 * x690
	assert set(x687.index_ids) == set(["i669","i636","i667","i670","i668",])
	assert x687.release_array("i669","i636","i667","i670","i668",).shape == (2,3,5,1,5,)
	assert x687.rank == 5
	x691 = NXArray(normal(size = (1,4,4,5,3,)), "i680","i676","i679","i677","i678",)
	x692 = NXArray(normal(size = (4,3,5,4,1,)), "i676","i678","i677","i679","i680",)
	x688 = x691 * x692
	assert set(x688.index_ids) == set([])
	assert x688.release_array().shape == ()
	assert x688.rank == 0
	x680 = x687 * x688
	assert set(x680.index_ids) == set(["i667","i670","i668","i669","i636",])
	assert x680.release_array("i667","i670","i668","i669","i636",).shape == (5,1,5,2,3,)
	assert x680.rank == 5
	x677 = x679 * x680
	assert set(x677.index_ids) == set(["i667","i636",])
	assert x677.release_array("i667","i636",).shape == (5,3,)
	assert x677.rank == 2
	x697 = NXArray(normal(size = (2,5,2,3,)), "i681","i682","i637","i505",)
	x698 = NXArray(normal(size = (2,5,5,2,5,)), "i681","i667","i682","i634","i635",)
	x695 = x697 * x698
	assert set(x695.index_ids) == set(["i505","i637","i634","i667","i635",])
	assert x695.release_array("i505","i637","i634","i667","i635",).shape == (3,2,2,5,5,)
	assert x695.rank == 5
	x699 = NXArray(normal(size = (2,4,3,1,1,)), "i683","i687","i684","i685","i686",)
	x700 = NXArray(normal(size = (3,2,1,1,4,)), "i684","i683","i685","i686","i687",)
	x696 = x699 * x700
	assert set(x696.index_ids) == set([])
	assert x696.release_array().shape == ()
	assert x696.rank == 0
	x693 = x695 * x696
	assert set(x693.index_ids) == set(["i505","i634","i667","i637","i635",])
	assert x693.release_array("i505","i634","i667","i637","i635",).shape == (3,2,5,2,5,)
	assert x693.rank == 5
	x703 = NXArray(normal(size = (3,4,5,1,4,)), "i693","i694","i692","i688","i691",)
	x704 = NXArray(normal(size = (4,3,5,3,)), "i694","i693","i690","i689",)
	x701 = x703 * x704
	assert set(x701.index_ids) == set(["i692","i688","i691","i690","i689",])
	assert x701.release_array("i692","i688","i691","i690","i689",).shape == (5,1,4,5,3,)
	assert x701.rank == 5
	x705 = NXArray(normal(size = ()), )
	x706 = NXArray(normal(size = (5,5,1,4,3,)), "i692","i690","i688","i691","i689",)
	x702 = x705 * x706
	assert set(x702.index_ids) == set(["i692","i690","i691","i688","i689",])
	assert x702.release_array("i692","i690","i691","i688","i689",).shape == (5,5,4,1,3,)
	assert x702.rank == 5
	x694 = x701 * x702
	assert set(x694.index_ids) == set([])
	assert x694.release_array().shape == ()
	assert x694.rank == 0
	x678 = x693 * x694
	assert set(x678.index_ids) == set(["i634","i667","i635","i505","i637",])
	assert x678.release_array("i634","i667","i635","i505","i637",).shape == (2,5,5,3,2,)
	assert x678.rank == 5
	x646 = x677 * x678
	assert set(x646.index_ids) == set(["i636","i635","i505","i634","i637",])
	assert x646.release_array("i636","i635","i505","i634","i637",).shape == (3,5,3,2,2,)
	assert x646.rank == 5
	x643 = x645 * x646
	assert set(x643.index_ids) == set(["i633","i505",])
	assert x643.release_array("i633","i505",).shape == (5,3,)
	assert x643.rank == 2
	x715 = NXArray(normal(size = (1,5,4,)), "i703","i705","i704",)
	x716 = NXArray(normal(size = (4,1,5,5,1,)), "i704","i703","i705","i702","i700",)
	x713 = x715 * x716
	assert set(x713.index_ids) == set(["i702","i700",])
	assert x713.release_array("i702","i700",).shape == (5,1,)
	assert x713.rank == 2
	x717 = NXArray(normal(size = (3,4,2,3,5,)), "i698","i699","i706","i707","i702",)
	x718 = NXArray(normal(size = (3,3,2,2,)), "i707","i696","i706","i701",)
	x714 = x717 * x718
	assert set(x714.index_ids) == set(["i702","i699","i698","i696","i701",])
	assert x714.release_array("i702","i699","i698","i696","i701",).shape == (5,4,3,3,2,)
	assert x714.rank == 5
	x711 = x713 * x714
	assert set(x711.index_ids) == set(["i700","i699","i698","i696","i701",])
	assert x711.release_array("i700","i699","i698","i696","i701",).shape == (1,4,3,3,2,)
	assert x711.rank == 5
	x721 = NXArray(normal(size = (1,3,1,2,1,)), "i710","i711","i712","i701","i709",)
	x722 = NXArray(normal(size = (1,1,3,4,)), "i710","i712","i711","i708",)
	x719 = x721 * x722
	assert set(x719.index_ids) == set(["i709","i701","i708",])
	assert x719.release_array("i709","i701","i708",).shape == (1,2,4,)
	assert x719.rank == 3
	x723 = NXArray(normal(size = (5,3,)), "i713","i697",)
	x724 = NXArray(normal(size = (5,1,4,4,1,)), "i713","i709","i708","i699","i700",)
	x720 = x723 * x724
	assert set(x720.index_ids) == set(["i697","i709","i699","i708","i700",])
	assert x720.release_array("i697","i709","i699","i708","i700",).shape == (3,1,4,4,1,)
	assert x720.rank == 5
	x712 = x719 * x720
	assert set(x712.index_ids) == set(["i701","i697","i700","i699",])
	assert x712.release_array("i701","i697","i700","i699",).shape == (2,3,1,4,)
	assert x712.rank == 4
	x709 = x711 * x712
	assert set(x709.index_ids) == set(["i696","i698","i697",])
	assert x709.release_array("i696","i698","i697",).shape == (3,3,3,)
	assert x709.rank == 3
	x729 = NXArray(normal(size = (5,1,1,3,1,)), "i715","i716","i718","i717","i714",)
	x730 = NXArray(normal(size = ()), )
	x727 = x729 * x730
	assert set(x727.index_ids) == set(["i718","i714","i716","i715","i717",])
	assert x727.release_array("i718","i714","i716","i715","i717",).shape == (1,1,1,5,3,)
	assert x727.rank == 5
	x731 = NXArray(normal(size = ()), )
	x732 = NXArray(normal(size = (1,5,1,1,3,)), "i718","i715","i714","i716","i717",)
	x728 = x731 * x732
	assert set(x728.index_ids) == set(["i718","i714","i715","i716","i717",])
	assert x728.release_array("i718","i714","i715","i716","i717",).shape == (1,1,5,1,3,)
	assert x728.rank == 5
	x725 = x727 * x728
	assert set(x725.index_ids) == set([])
	assert x725.release_array().shape == ()
	assert x725.rank == 0
	x735 = NXArray(normal(size = ()), )
	x736 = NXArray(normal(size = (4,2,1,1,3,)), "i506","i719","i720","i695","i697",)
	x733 = x735 * x736
	assert set(x733.index_ids) == set(["i695","i720","i506","i719","i697",])
	assert x733.release_array("i695","i720","i506","i719","i697",).shape == (1,1,4,2,3,)
	assert x733.rank == 5
	x737 = NXArray(normal(size = (5,5,2,)), "i722","i721","i719",)
	x738 = NXArray(normal(size = (3,5,1,5,3,)), "i696","i721","i720","i722","i698",)
	x734 = x737 * x738
	assert set(x734.index_ids) == set(["i719","i696","i720","i698",])
	assert x734.release_array("i719","i696","i720","i698",).shape == (2,3,1,3,)
	assert x734.rank == 4
	x726 = x733 * x734
	assert set(x726.index_ids) == set(["i695","i697","i506","i698","i696",])
	assert x726.release_array("i695","i697","i506","i698","i696",).shape == (1,3,4,3,3,)
	assert x726.rank == 5
	x710 = x725 * x726
	assert set(x710.index_ids) == set(["i506","i698","i697","i695","i696",])
	assert x710.release_array("i506","i698","i697","i695","i696",).shape == (4,3,3,1,3,)
	assert x710.rank == 5
	x707 = x709 * x710
	assert set(x707.index_ids) == set(["i695","i506",])
	assert x707.release_array("i695","i506",).shape == (1,4,)
	assert x707.rank == 2
	x745 = NXArray(normal(size = (5,3,4,1,2,)), "i730","i726","i727","i725","i731",)
	x746 = NXArray(normal(size = (2,2,3,5,)), "i729","i731","i728","i730",)
	x743 = x745 * x746
	assert set(x743.index_ids) == set(["i726","i725","i727","i728","i729",])
	assert x743.release_array("i726","i725","i727","i728","i729",).shape == (3,1,4,3,2,)
	assert x743.rank == 5
	x747 = NXArray(normal(size = (3,)), "i732",)
	x748 = NXArray(normal(size = (3,4,3,3,2,)), "i728","i727","i732","i726","i729",)
	x744 = x747 * x748
	assert set(x744.index_ids) == set(["i727","i726","i728","i729",])
	assert x744.release_array("i727","i726","i728","i729",).shape == (4,3,3,2,)
	assert x744.rank == 4
	x741 = x743 * x744
	assert set(x741.index_ids) == set(["i725",])
	assert x741.release_array("i725",).shape == (1,)
	assert x741.rank == 1
	x751 = NXArray(normal(size = (2,2,3,2,1,)), "i737","i733","i735","i734","i736",)
	x752 = NXArray(normal(size = (2,2,3,2,1,)), "i733","i737","i735","i734","i736",)
	x749 = x751 * x752
	assert set(x749.index_ids) == set([])
	assert x749.release_array().shape == ()
	assert x749.rank == 0
	x753 = NXArray(normal(size = (2,1,2,1,)), "i739","i695","i738","i725",)
	x754 = NXArray(normal(size = (3,2,2,5,5,)), "i4","i738","i739","i724","i723",)
	x750 = x753 * x754
	assert set(x750.index_ids) == set(["i695","i725","i724","i723","i4",])
	assert x750.release_array("i695","i725","i724","i723","i4",).shape == (1,1,5,5,3,)
	assert x750.rank == 5
	x742 = x749 * x750
	assert set(x742.index_ids) == set(["i725","i695","i724","i723","i4",])
	assert x742.release_array("i725","i695","i724","i723","i4",).shape == (1,1,5,5,3,)
	assert x742.rank == 5
	x739 = x741 * x742
	assert set(x739.index_ids) == set(["i723","i724","i4","i695",])
	assert x739.release_array("i723","i724","i4","i695",).shape == (5,5,3,1,)
	assert x739.rank == 4
	x759 = NXArray(normal(size = (1,4,4,5,1,)), "i741","i742","i743","i744","i740",)
	x760 = NXArray(normal(size = (4,5,4,5,)), "i6","i744","i743","i723",)
	x757 = x759 * x760
	assert set(x757.index_ids) == set(["i740","i742","i741","i6","i723",])
	assert x757.release_array("i740","i742","i741","i6","i723",).shape == (1,4,1,4,5,)
	assert x757.rank == 5
	x761 = NXArray(normal(size = (5,5,5,)), "i747","i745","i746",)
	x762 = NXArray(normal(size = (5,5,4,5,5,)), "i746","i633","i742","i747","i745",)
	x758 = x761 * x762
	assert set(x758.index_ids) == set(["i742","i633",])
	assert x758.release_array("i742","i633",).shape == (4,5,)
	assert x758.rank == 2
	x755 = x757 * x758
	assert set(x755.index_ids) == set(["i741","i740","i6","i723","i633",])
	assert x755.release_array("i741","i740","i6","i723","i633",).shape == (1,1,4,5,5,)
	assert x755.rank == 5
	x765 = NXArray(normal(size = (1,3,4,5,2,)), "i740","i751","i748","i724","i749",)
	x766 = NXArray(normal(size = (3,3,)), "i751","i750",)
	x763 = x765 * x766
	assert set(x763.index_ids) == set(["i740","i748","i749","i724","i750",])
	assert x763.release_array("i740","i748","i749","i724","i750",).shape == (1,4,2,5,3,)
	assert x763.rank == 5
	x767 = NXArray(normal(size = (4,3,3,5,5,)), "i748","i5","i750","i752","i753",)
	x768 = NXArray(normal(size = (5,5,1,2,)), "i752","i753","i741","i749",)
	x764 = x767 * x768
	assert set(x764.index_ids) == set(["i748","i5","i750","i741","i749",])
	assert x764.release_array("i748","i5","i750","i741","i749",).shape == (4,3,3,1,2,)
	assert x764.rank == 5
	x756 = x763 * x764
	assert set(x756.index_ids) == set(["i740","i724","i741","i5",])
	assert x756.release_array("i740","i724","i741","i5",).shape == (1,5,1,3,)
	assert x756.rank == 4
	x740 = x755 * x756
	assert set(x740.index_ids) == set(["i723","i633","i6","i724","i5",])
	assert x740.release_array("i723","i633","i6","i724","i5",).shape == (5,5,4,5,3,)
	assert x740.rank == 5
	x708 = x739 * x740
	assert set(x708.index_ids) == set(["i695","i4","i633","i6","i5",])
	assert x708.release_array("i695","i4","i633","i6","i5",).shape == (1,3,5,4,3,)
	assert x708.rank == 5
	x644 = x707 * x708
	assert set(x644.index_ids) == set(["i506","i633","i6","i5","i4",])
	assert x644.release_array("i506","i633","i6","i5","i4",).shape == (4,5,4,3,3,)
	assert x644.rank == 5
	x516 = x643 * x644
	assert set(x516.index_ids) == set(["i505","i4","i506","i5","i6",])
	assert x516.release_array("i505","i4","i506","i5","i6",).shape == (3,3,4,3,4,)
	assert x516.rank == 5
	x513 = x515 * x516
	assert set(x513.index_ids) == set(["i503","i504","i6","i5","i4",])
	assert x513.release_array("i503","i504","i6","i5","i4",).shape == (5,4,4,3,3,)
	assert x513.rank == 5
	x781 = NXArray(normal(size = (2,4,2,1,5,)), "i763","i767","i766","i765","i764",)
	x782 = NXArray(normal(size = (4,2,)), "i767","i762",)
	x779 = x781 * x782
	assert set(x779.index_ids) == set(["i766","i765","i764","i763","i762",])
	assert x779.release_array("i766","i765","i764","i763","i762",).shape == (2,1,5,2,2,)
	assert x779.rank == 5
	x783 = NXArray(normal(size = (2,1,2,1,5,)), "i766","i765","i763","i761","i764",)
	x784 = NXArray(normal(size = ()), )
	x780 = x783 * x784
	assert set(x780.index_ids) == set(["i763","i761","i764","i765","i766",])
	assert x780.release_array("i763","i761","i764","i765","i766",).shape == (2,1,5,1,2,)
	assert x780.rank == 5
	x777 = x779 * x780
	assert set(x777.index_ids) == set(["i762","i761",])
	assert x777.release_array("i762","i761",).shape == (2,1,)
	assert x777.rank == 2
	x787 = NXArray(normal(size = (2,5,2,2,2,)), "i771","i770","i768","i769","i772",)
	x788 = NXArray(normal(size = (2,2,2,5,2,)), "i771","i768","i769","i770","i772",)
	x785 = x787 * x788
	assert set(x785.index_ids) == set([])
	assert x785.release_array().shape == ()
	assert x785.rank == 0
	x789 = NXArray(normal(size = (2,1,5,5,5,)), "i759","i756","i774","i773","i3",)
	x790 = NXArray(normal(size = (5,2,2,5,)), "i773","i760","i762","i774",)
	x786 = x789 * x790
	assert set(x786.index_ids) == set(["i3","i756","i759","i762","i760",])
	assert x786.release_array("i3","i756","i759","i762","i760",).shape == (5,1,2,2,2,)
	assert x786.rank == 5
	x778 = x785 * x786
	assert set(x778.index_ids) == set(["i756","i760","i3","i762","i759",])
	assert x778.release_array("i756","i760","i3","i762","i759",).shape == (1,2,5,2,2,)
	assert x778.rank == 5
	x775 = x777 * x778
	assert set(x775.index_ids) == set(["i761","i756","i3","i760","i759",])
	assert x775.release_array("i761","i756","i3","i760","i759",).shape == (1,1,5,2,2,)
	assert x775.rank == 5
	x795 = NXArray(normal(size = ()), )
	x796 = NXArray(normal(size = (3,5,2,1,3,)), "i779","i778","i776","i777","i775",)
	x793 = x795 * x796
	assert set(x793.index_ids) == set(["i778","i779","i776","i777","i775",])
	assert x793.release_array("i778","i779","i776","i777","i775",).shape == (5,3,2,1,3,)
	assert x793.rank == 5
	x797 = NXArray(normal(size = (2,3,1,)), "i781","i780","i782",)
	x798 = NXArray(normal(size = (3,2,3,5,1,)), "i780","i781","i779","i778","i782",)
	x794 = x797 * x798
	assert set(x794.index_ids) == set(["i778","i779",])
	assert x794.release_array("i778","i779",).shape == (5,3,)
	assert x794.rank == 2
	x791 = x793 * x794
	assert set(x791.index_ids) == set(["i775","i776","i777",])
	assert x791.release_array("i775","i776","i777",).shape == (3,2,1,)
	assert x791.rank == 3
	x801 = NXArray(normal(size = (4,2,)), "i783","i784",)
	x802 = NXArray(normal(size = (2,3,2,2,1,)), "i784","i775","i757","i776","i761",)
	x799 = x801 * x802
	assert set(x799.index_ids) == set(["i783","i757","i775","i776","i761",])
	assert x799.release_array("i783","i757","i775","i776","i761",).shape == (4,2,3,2,1,)
	assert x799.rank == 5
	x803 = NXArray(normal(size = (3,1,1,1,4,)), "i786","i777","i785","i787","i783",)
	x804 = NXArray(normal(size = (1,3,1,)), "i785","i786","i787",)
	x800 = x803 * x804
	assert set(x800.index_ids) == set(["i783","i777",])
	assert x800.release_array("i783","i777",).shape == (4,1,)
	assert x800.rank == 2
	x792 = x799 * x800
	assert set(x792.index_ids) == set(["i757","i775","i776","i761","i777",])
	assert x792.release_array("i757","i775","i776","i761","i777",).shape == (2,3,2,1,1,)
	assert x792.rank == 5
	x776 = x791 * x792
	assert set(x776.index_ids) == set(["i757","i761",])
	assert x776.release_array("i757","i761",).shape == (2,1,)
	assert x776.rank == 2
	x773 = x775 * x776
	assert set(x773.index_ids) == set(["i3","i760","i756","i759","i757",])
	assert x773.release_array("i3","i760","i756","i759","i757",).shape == (5,2,1,2,2,)
	assert x773.rank == 5
	x811 = NXArray(normal(size = (1,2,4,2,2,)), "i791","i792","i793","i795","i794",)
	x812 = NXArray(normal(size = (4,2,1,2,2,)), "i793","i795","i791","i792","i794",)
	x809 = x811 * x812
	assert set(x809.index_ids) == set([])
	assert x809.release_array().shape == ()
	assert x809.rank == 0
	x813 = NXArray(normal(size = ()), )
	x814 = NXArray(normal(size = (2,2,4,3,2,)), "i760","i790","i789","i788","i759",)
	x810 = x813 * x814
	assert set(x810.index_ids) == set(["i790","i789","i788","i759","i760",])
	assert x810.release_array("i790","i789","i788","i759","i760",).shape == (2,4,3,2,2,)
	assert x810.rank == 5
	x807 = x809 * x810
	assert set(x807.index_ids) == set(["i760","i788","i790","i789","i759",])
	assert x807.release_array("i760","i788","i790","i789","i759",).shape == (2,3,2,4,2,)
	assert x807.rank == 5
	x817 = NXArray(normal(size = (4,1,2,4,3,)), "i799","i798","i790","i796","i797",)
	x818 = NXArray(normal(size = ()), )
	x815 = x817 * x818
	assert set(x815.index_ids) == set(["i799","i797","i796","i790","i798",])
	assert x815.release_array("i799","i797","i796","i790","i798",).shape == (4,3,4,2,1,)
	assert x815.rank == 5
	x819 = NXArray(normal(size = (1,4,3,4,4,)), "i798","i755","i797","i796","i799",)
	x820 = NXArray(normal(size = ()), )
	x816 = x819 * x820
	assert set(x816.index_ids) == set(["i799","i755","i796","i797","i798",])
	assert x816.release_array("i799","i755","i796","i797","i798",).shape == (4,4,4,3,1,)
	assert x816.rank == 5
	x808 = x815 * x816
	assert set(x808.index_ids) == set(["i790","i755",])
	assert x808.release_array("i790","i755",).shape == (2,4,)
	assert x808.rank == 2
	x805 = x807 * x808
	assert set(x805.index_ids) == set(["i788","i789","i760","i759","i755",])
	assert x805.release_array("i788","i789","i760","i759","i755",).shape == (3,4,2,2,4,)
	assert x805.rank == 5
	x825 = NXArray(normal(size = (5,5,)), "i801","i805",)
	x826 = NXArray(normal(size = (2,4,1,5,4,)), "i804","i800","i803","i805","i789",)
	x823 = x825 * x826
	assert set(x823.index_ids) == set(["i801","i800","i789","i803","i804",])
	assert x823.release_array("i801","i800","i789","i803","i804",).shape == (5,4,4,1,2,)
	assert x823.rank == 5
	x827 = NXArray(normal(size = (1,3,1,)), "i806","i807","i803",)
	x828 = NXArray(normal(size = (3,2,3,4,1,)), "i807","i804","i788","i802","i806",)
	x824 = x827 * x828
	assert set(x824.index_ids) == set(["i803","i788","i804","i802",])
	assert x824.release_array("i803","i788","i804","i802",).shape == (1,3,2,4,)
	assert x824.rank == 4
	x821 = x823 * x824
	assert set(x821.index_ids) == set(["i789","i801","i800","i788","i802",])
	assert x821.release_array("i789","i801","i800","i788","i802",).shape == (4,5,4,3,4,)
	assert x821.rank == 5
	x831 = NXArray(normal(size = (2,3,5,4,3,)), "i809","i808","i812","i810","i811",)
	x832 = NXArray(normal(size = (5,2,4,3,)), "i812","i809","i810","i811",)
	x829 = x831 * x832
	assert set(x829.index_ids) == set(["i808",])
	assert x829.release_array("i808",).shape == (3,)
	assert x829.rank == 1
	x833 = NXArray(normal(size = (3,4,5,5,4,)), "i808","i800","i758","i801","i802",)
	x834 = NXArray(normal(size = ()), )
	x830 = x833 * x834
	assert set(x830.index_ids) == set(["i808","i800","i802","i801","i758",])
	assert x830.release_array("i808","i800","i802","i801","i758",).shape == (3,4,4,5,5,)
	assert x830.rank == 5
	x822 = x829 * x830
	assert set(x822.index_ids) == set(["i802","i801","i800","i758",])
	assert x822.release_array("i802","i801","i800","i758",).shape == (4,5,4,5,)
	assert x822.rank == 4
	x806 = x821 * x822
	assert set(x806.index_ids) == set(["i788","i789","i758",])
	assert x806.release_array("i788","i789","i758",).shape == (3,4,5,)
	assert x806.rank == 3
	x774 = x805 * x806
	assert set(x774.index_ids) == set(["i760","i755","i759","i758",])
	assert x774.release_array("i760","i755","i759","i758",).shape == (2,4,2,5,)
	assert x774.rank == 4
	x771 = x773 * x774
	assert set(x771.index_ids) == set(["i3","i756","i757","i758","i755",])
	assert x771.release_array("i3","i756","i757","i758","i755",).shape == (5,1,2,5,4,)
	assert x771.rank == 5
	x843 = NXArray(normal(size = (4,3,4,1,)), "i819","i815","i817","i820",)
	x844 = NXArray(normal(size = (2,5,1,4,3,)), "i816","i818","i820","i819","i814",)
	x841 = x843 * x844
	assert set(x841.index_ids) == set(["i817","i815","i818","i816","i814",])
	assert x841.release_array("i817","i815","i818","i816","i814",).shape == (4,3,5,2,3,)
	assert x841.rank == 5
	x845 = NXArray(normal(size = (2,2,3,4,3,)), "i821","i816","i822","i817","i815",)
	x846 = NXArray(normal(size = (4,2,3,5,)), "i504","i821","i822","i818",)
	x842 = x845 * x846
	assert set(x842.index_ids) == set(["i817","i815","i816","i818","i504",])
	assert x842.release_array("i817","i815","i816","i818","i504",).shape == (4,3,2,5,4,)
	assert x842.rank == 5
	x839 = x841 * x842
	assert set(x839.index_ids) == set(["i814","i504",])
	assert x839.release_array("i814","i504",).shape == (3,4,)
	assert x839.rank == 2
	x849 = NXArray(normal(size = (4,3,)), "i825","i814",)
	x850 = NXArray(normal(size = (4,4,1,3,5,)), "i825","i824","i813","i823","i758",)
	x847 = x849 * x850
	assert set(x847.index_ids) == set(["i814","i824","i813","i823","i758",])
	assert x847.release_array("i814","i824","i813","i823","i758",).shape == (3,4,1,3,5,)
	assert x847.rank == 5
	x851 = NXArray(normal(size = (3,4,2,2,4,)), "i823","i826","i754","i827","i824",)
	x852 = NXArray(normal(size = (2,2,4,)), "i827","i757","i826",)
	x848 = x851 * x852
	assert set(x848.index_ids) == set(["i754","i824","i823","i757",])
	assert x848.release_array("i754","i824","i823","i757",).shape == (2,4,3,2,)
	assert x848.rank == 4
	x840 = x847 * x848
	assert set(x840.index_ids) == set(["i814","i813","i758","i757","i754",])
	assert x840.release_array("i814","i813","i758","i757","i754",).shape == (3,1,5,2,2,)
	assert x840.rank == 5
	x837 = x839 * x840
	assert set(x837.index_ids) == set(["i504","i754","i757","i813","i758",])
	assert x837.release_array("i504","i754","i757","i813","i758",).shape == (4,2,2,1,5,)
	assert x837.rank == 5
	x857 = NXArray(normal(size = (1,2,3,2,4,)), "i833","i837","i834","i835","i836",)
	x858 = NXArray(normal(size = (3,1,2,4,2,)), "i834","i833","i837","i836","i835",)
	x855 = x857 * x858
	assert set(x855.index_ids) == set([])
	assert x855.release_array().shape == ()
	assert x855.rank == 0
	x859 = NXArray(normal(size = (3,4,4,5,)), "i838","i829","i830","i839",)
	x860 = NXArray(normal(size = (4,5,3,5,1,)), "i831","i832","i838","i839","i828",)
	x856 = x859 * x860
	assert set(x856.index_ids) == set(["i829","i830","i828","i831","i832",])
	assert x856.release_array("i829","i830","i828","i831","i832",).shape == (4,4,1,4,5,)
	assert x856.rank == 5
	x853 = x855 * x856
	assert set(x853.index_ids) == set(["i829","i828","i832","i831","i830",])
	assert x853.release_array("i829","i828","i832","i831","i830",).shape == (4,1,5,4,4,)
	assert x853.rank == 5
	x863 = NXArray(normal(size = (4,3,1,5,2,)), "i844","i840","i841","i842","i843",)
	x864 = NXArray(normal(size = (2,3,1,4,5,)), "i843","i840","i841","i844","i842",)
	x861 = x863 * x864
	assert set(x861.index_ids) == set([])
	assert x861.release_array().shape == ()
	assert x861.rank == 0
	x865 = NXArray(normal(size = (1,4,1,5,4,)), "i845","i831","i828","i832","i829",)
	x866 = NXArray(normal(size = (4,1,)), "i830","i845",)
	x862 = x865 * x866
	assert set(x862.index_ids) == set(["i832","i829","i828","i831","i830",])
	assert x862.release_array("i832","i829","i828","i831","i830",).shape == (5,4,1,4,4,)
	assert x862.rank == 5
	x854 = x861 * x862
	assert set(x854.index_ids) == set(["i829","i828","i830","i831","i832",])
	assert x854.release_array("i829","i828","i830","i831","i832",).shape == (4,1,4,4,5,)
	assert x854.rank == 5
	x838 = x853 * x854
	assert set(x838.index_ids) == set([])
	assert x838.release_array().shape == ()
	assert x838.rank == 0
	x835 = x837 * x838
	assert set(x835.index_ids) == set(["i754","i758","i813","i757","i504",])
	assert x835.release_array("i754","i758","i813","i757","i504",).shape == (2,5,1,2,4,)
	assert x835.rank == 5
	x873 = NXArray(normal(size = (4,1,5,2,4,)), "i852","i853","i851","i854","i855",)
	x874 = NXArray(normal(size = (4,2,)), "i855","i854",)
	x871 = x873 * x874
	assert set(x871.index_ids) == set(["i853","i851","i852",])
	assert x871.release_array("i853","i851","i852",).shape == (1,5,4,)
	assert x871.rank == 3
	x875 = NXArray(normal(size = (3,5,)), "i849","i856",)
	x876 = NXArray(normal(size = (5,4,1,5,4,)), "i851","i852","i853","i856","i850",)
	x872 = x875 * x876
	assert set(x872.index_ids) == set(["i849","i853","i852","i850","i851",])
	assert x872.release_array("i849","i853","i852","i850","i851",).shape == (3,1,4,4,5,)
	assert x872.rank == 5
	x869 = x871 * x872
	assert set(x869.index_ids) == set(["i850","i849",])
	assert x869.release_array("i850","i849",).shape == (4,3,)
	assert x869.rank == 2
	x879 = NXArray(normal(size = (3,4,5,2,1,)), "i859","i850","i858","i860","i857",)
	x880 = NXArray(normal(size = (2,5,3,)), "i860","i858","i859",)
	x877 = x879 * x880
	assert set(x877.index_ids) == set(["i857","i850",])
	assert x877.release_array("i857","i850",).shape == (1,4,)
	assert x877.rank == 2
	x881 = NXArray(normal(size = (5,2,1,1,5,)), "i846","i848","i857","i813","i847",)
	x882 = NXArray(normal(size = ()), )
	x878 = x881 * x882
	assert set(x878.index_ids) == set(["i846","i848","i857","i813","i847",])
	assert x878.release_array("i846","i848","i857","i813","i847",).shape == (5,2,1,1,5,)
	assert x878.rank == 5
	x870 = x877 * x878
	assert set(x870.index_ids) == set(["i850","i813","i847","i846","i848",])
	assert x870.release_array("i850","i813","i847","i846","i848",).shape == (4,1,5,5,2,)
	assert x870.rank == 5
	x867 = x869 * x870
	assert set(x867.index_ids) == set(["i849","i846","i847","i813","i848",])
	assert x867.release_array("i849","i846","i847","i813","i848",).shape == (3,5,5,1,2,)
	assert x867.rank == 5
	x887 = NXArray(normal(size = ()), )
	x888 = NXArray(normal(size = (2,5,3,5,5,)), "i848","i847","i849","i846","i862",)
	x885 = x887 * x888
	assert set(x885.index_ids) == set(["i848","i862","i847","i849","i846",])
	assert x885.release_array("i848","i862","i847","i849","i846",).shape == (2,5,5,3,5,)
	assert x885.rank == 5
	x889 = NXArray(normal(size = (1,4,4,)), "i865","i863","i864",)
	x890 = NXArray(normal(size = (4,4,1,1,5,)), "i863","i864","i861","i865","i862",)
	x886 = x889 * x890
	assert set(x886.index_ids) == set(["i862","i861",])
	assert x886.release_array("i862","i861",).shape == (5,1,)
	assert x886.rank == 2
	x883 = x885 * x886
	assert set(x883.index_ids) == set(["i846","i848","i847","i849","i861",])
	assert x883.release_array("i846","i848","i847","i849","i861",).shape == (5,2,5,3,1,)
	assert x883.rank == 5
	x893 = NXArray(normal(size = (3,5,2,2,1,)), "i867","i869","i866","i868","i861",)
	x894 = NXArray(normal(size = ()), )
	x891 = x893 * x894
	assert set(x891.index_ids) == set(["i869","i866","i868","i867","i861",])
	assert x891.release_array("i869","i866","i868","i867","i861",).shape == (5,2,2,3,1,)
	assert x891.rank == 5
	x895 = NXArray(normal(size = (2,2,2,5,3,)), "i866","i870","i868","i871","i867",)
	x896 = NXArray(normal(size = (5,2,5,)), "i869","i870","i871",)
	x892 = x895 * x896
	assert set(x892.index_ids) == set(["i867","i866","i868","i869",])
	assert x892.release_array("i867","i866","i868","i869",).shape == (3,2,2,5,)
	assert x892.rank == 4
	x884 = x891 * x892
	assert set(x884.index_ids) == set(["i861",])
	assert x884.release_array("i861",).shape == (1,)
	assert x884.rank == 1
	x868 = x883 * x884
	assert set(x868.index_ids) == set(["i847","i846","i849","i848",])
	assert x868.release_array("i847","i846","i849","i848",).shape == (5,5,3,2,)
	assert x868.rank == 4
	x836 = x867 * x868
	assert set(x836.index_ids) == set(["i813",])
	assert x836.release_array("i813",).shape == (1,)
	assert x836.rank == 1
	x772 = x835 * x836
	assert set(x772.index_ids) == set(["i504","i754","i757","i758",])
	assert x772.release_array("i504","i754","i757","i758",).shape == (4,2,2,5,)
	assert x772.rank == 4
	x769 = x771 * x772
	assert set(x769.index_ids) == set(["i3","i756","i755","i754","i504",])
	assert x769.release_array("i3","i756","i755","i754","i504",).shape == (5,1,4,2,4,)
	assert x769.rank == 5
	x907 = NXArray(normal(size = (3,4,2,2,5,)), "i883","i882","i881","i879","i876",)
	x908 = NXArray(normal(size = (3,5,5,2,4,)), "i883","i875","i880","i881","i882",)
	x905 = x907 * x908
	assert set(x905.index_ids) == set(["i879","i876","i880","i875",])
	assert x905.release_array("i879","i876","i880","i875",).shape == (2,5,5,5,)
	assert x905.rank == 4
	x909 = NXArray(normal(size = (5,2,4,2,4,)), "i880","i878","i877","i879","i874",)
	x910 = NXArray(normal(size = ()), )
	x906 = x909 * x910
	assert set(x906.index_ids) == set(["i878","i879","i874","i880","i877",])
	assert x906.release_array("i878","i879","i874","i880","i877",).shape == (2,2,4,5,4,)
	assert x906.rank == 5
	x903 = x905 * x906
	assert set(x903.index_ids) == set(["i876","i875","i874","i878","i877",])
	assert x903.release_array("i876","i875","i874","i878","i877",).shape == (5,5,4,2,4,)
	assert x903.rank == 5
	x913 = NXArray(normal(size = (3,2,2,5,1,)), "i885","i887","i878","i888","i884",)
	x914 = NXArray(normal(size = (5,1,)), "i888","i886",)
	x911 = x913 * x914
	assert set(x911.index_ids) == set(["i885","i878","i887","i884","i886",])
	assert x911.release_array("i885","i878","i887","i884","i886",).shape == (3,2,2,1,1,)
	assert x911.rank == 5
	x915 = NXArray(normal(size = (4,3,4,1,)), "i877","i889","i890","i884",)
	x916 = NXArray(normal(size = (1,2,3,3,4,)), "i886","i887","i885","i889","i890",)
	x912 = x915 * x916
	assert set(x912.index_ids) == set(["i884","i877","i885","i886","i887",])
	assert x912.release_array("i884","i877","i885","i886","i887",).shape == (1,4,3,1,2,)
	assert x912.rank == 5
	x904 = x911 * x912
	assert set(x904.index_ids) == set(["i878","i877",])
	assert x904.release_array("i878","i877",).shape == (2,4,)
	assert x904.rank == 2
	x901 = x903 * x904
	assert set(x901.index_ids) == set(["i874","i876","i875",])
	assert x901.release_array("i874","i876","i875",).shape == (4,5,5,)
	assert x901.rank == 3
	x921 = NXArray(normal(size = ()), )
	x922 = NXArray(normal(size = (2,2,5,5,4,)), "i891","i872","i875","i876","i874",)
	x919 = x921 * x922
	assert set(x919.index_ids) == set(["i872","i875","i891","i876","i874",])
	assert x919.release_array("i872","i875","i891","i876","i874",).shape == (2,5,2,5,4,)
	assert x919.rank == 5
	x923 = NXArray(normal(size = (2,3,2,)), "i892","i894","i893",)
	x924 = NXArray(normal(size = (3,2,2,3,2,)), "i894","i893","i892","i873","i891",)
	x920 = x923 * x924
	assert set(x920.index_ids) == set(["i891","i873",])
	assert x920.release_array("i891","i873",).shape == (2,3,)
	assert x920.rank == 2
	x917 = x919 * x920
	assert set(x917.index_ids) == set(["i875","i876","i872","i874","i873",])
	assert x917.release_array("i875","i876","i872","i874","i873",).shape == (5,5,2,4,3,)
	assert x917.rank == 5
	x927 = NXArray(normal(size = (2,5,4,2,2,)), "i900","i901","i897","i895","i898",)
	x928 = NXArray(normal(size = (2,2,2,5,)), "i896","i899","i900","i901",)
	x925 = x927 * x928
	assert set(x925.index_ids) == set(["i897","i898","i895","i899","i896",])
	assert x925.release_array("i897","i898","i895","i899","i896",).shape == (4,2,2,2,2,)
	assert x925.rank == 5
	x929 = NXArray(normal(size = (2,1,)), "i895","i902",)
	x930 = NXArray(normal(size = (2,2,4,2,1,)), "i899","i896","i897","i898","i902",)
	x926 = x929 * x930
	assert set(x926.index_ids) == set(["i895","i897","i899","i898","i896",])
	assert x926.release_array("i895","i897","i899","i898","i896",).shape == (2,4,2,2,2,)
	assert x926.rank == 5
	x918 = x925 * x926
	assert set(x918.index_ids) == set([])
	assert x918.release_array().shape == ()
	assert x918.rank == 0
	x902 = x917 * x918
	assert set(x902.index_ids) == set(["i875","i874","i876","i872","i873",])
	assert x902.release_array("i875","i874","i876","i872","i873",).shape == (5,4,5,2,3,)
	assert x902.rank == 5
	x899 = x901 * x902
	assert set(x899.index_ids) == set(["i873","i872",])
	assert x899.release_array("i873","i872",).shape == (3,2,)
	assert x899.rank == 2
	x937 = NXArray(normal(size = (5,5,5,3,4,)), "i906","i904","i903","i905","i907",)
	x938 = NXArray(normal(size = ()), )
	x935 = x937 * x938
	assert set(x935.index_ids) == set(["i906","i903","i905","i904","i907",])
	assert x935.release_array("i906","i903","i905","i904","i907",).shape == (5,5,3,5,4,)
	assert x935.rank == 5
	x939 = NXArray(normal(size = (5,5,5,4,3,)), "i903","i906","i904","i907","i905",)
	x940 = NXArray(normal(size = ()), )
	x936 = x939 * x940
	assert set(x936.index_ids) == set(["i906","i904","i903","i905","i907",])
	assert x936.release_array("i906","i904","i903","i905","i907",).shape == (5,5,5,3,4,)
	assert x936.rank == 5
	x933 = x935 * x936
	assert set(x933.index_ids) == set([])
	assert x933.release_array().shape == ()
	assert x933.rank == 0
	x943 = NXArray(normal(size = ()), )
	x944 = NXArray(normal(size = (2,4,3,4,5,)), "i754","i755","i873","i908","i503",)
	x941 = x943 * x944
	assert set(x941.index_ids) == set(["i908","i755","i754","i873","i503",])
	assert x941.release_array("i908","i755","i754","i873","i503",).shape == (4,4,2,3,5,)
	assert x941.rank == 5
	x945 = NXArray(normal(size = (5,4,4,3,4,)), "i909","i908","i911","i910","i912",)
	x946 = NXArray(normal(size = (4,5,3,1,4,)), "i911","i909","i910","i756","i912",)
	x942 = x945 * x946
	assert set(x942.index_ids) == set(["i908","i756",])
	assert x942.release_array("i908","i756",).shape == (4,1,)
	assert x942.rank == 2
	x934 = x941 * x942
	assert set(x934.index_ids) == set(["i755","i754","i873","i503","i756",])
	assert x934.release_array("i755","i754","i873","i503","i756",).shape == (4,2,3,5,1,)
	assert x934.rank == 5
	x931 = x933 * x934
	assert set(x931.index_ids) == set(["i755","i754","i873","i503","i756",])
	assert x931.release_array("i755","i754","i873","i503","i756",).shape == (4,2,3,5,1,)
	assert x931.rank == 5
	x951 = NXArray(normal(size = (5,1,1,2,)), "i919","i918","i916","i913",)
	x952 = NXArray(normal(size = (5,1,5,4,2,)), "i919","i918","i917","i915","i914",)
	x949 = x951 * x952
	assert set(x949.index_ids) == set(["i916","i913","i917","i915","i914",])
	assert x949.release_array("i916","i913","i917","i915","i914",).shape == (1,2,5,4,2,)
	assert x949.rank == 5
	x953 = NXArray(normal(size = (3,2,4,4,1,)), "i924","i923","i920","i921","i922",)
	x954 = NXArray(normal(size = (3,4,4,2,1,)), "i924","i921","i920","i923","i922",)
	x950 = x953 * x954
	assert set(x950.index_ids) == set([])
	assert x950.release_array().shape == ()
	assert x950.rank == 0
	x947 = x949 * x950
	assert set(x947.index_ids) == set(["i916","i917","i913","i915","i914",])
	assert x947.release_array("i916","i917","i913","i915","i914",).shape == (1,5,2,4,2,)
	assert x947.rank == 5
	x957 = NXArray(normal(size = (4,2,4,)), "i926","i927","i928",)
	x958 = NXArray(normal(size = (4,2,4,3,2,)), "i926","i927","i928","i925","i914",)
	x955 = x957 * x958
	assert set(x955.index_ids) == set(["i914","i925",])
	assert x955.release_array("i914","i925",).shape == (2,3,)
	assert x955.rank == 2
	x959 = NXArray(normal(size = (3,2,)), "i929","i913",)
	x960 = NXArray(normal(size = (3,4,1,5,3,)), "i929","i915","i916","i917","i925",)
	x956 = x959 * x960
	assert set(x956.index_ids) == set(["i913","i915","i916","i917","i925",])
	assert x956.release_array("i913","i915","i916","i917","i925",).shape == (2,4,1,5,3,)
	assert x956.rank == 5
	x948 = x955 * x956
	assert set(x948.index_ids) == set(["i914","i917","i915","i913","i916",])
	assert x948.release_array("i914","i917","i915","i913","i916",).shape == (2,5,4,2,1,)
	assert x948.rank == 5
	x932 = x947 * x948
	assert set(x932.index_ids) == set([])
	assert x932.release_array().shape == ()
	assert x932.rank == 0
	x900 = x931 * x932
	assert set(x900.index_ids) == set(["i873","i503","i754","i756","i755",])
	assert x900.release_array("i873","i503","i754","i756","i755",).shape == (3,5,2,1,4,)
	assert x900.rank == 5
	x897 = x899 * x900
	assert set(x897.index_ids) == set(["i872","i755","i503","i756","i754",])
	assert x897.release_array("i872","i755","i503","i756","i754",).shape == (2,4,5,1,2,)
	assert x897.rank == 5
	x969 = NXArray(normal(size = (2,5,1,4,2,)), "i940","i936","i937","i935","i939",)
	x970 = NXArray(normal(size = (2,3,)), "i940","i938",)
	x967 = x969 * x970
	assert set(x967.index_ids) == set(["i936","i937","i935","i939","i938",])
	assert x967.release_array("i936","i937","i935","i939","i938",).shape == (5,1,4,2,3,)
	assert x967.rank == 5
	x971 = NXArray(normal(size = ()), )
	x972 = NXArray(normal(size = (4,1,5,3,2,)), "i935","i937","i936","i938","i939",)
	x968 = x971 * x972
	assert set(x968.index_ids) == set(["i935","i937","i936","i938","i939",])
	assert x968.release_array("i935","i937","i936","i938","i939",).shape == (4,1,5,3,2,)
	assert x968.rank == 5
	x965 = x967 * x968
	assert set(x965.index_ids) == set([])
	assert x965.release_array().shape == ()
	assert x965.rank == 0
	x975 = NXArray(normal(size = (4,5,1,3,3,)), "i941","i943","i942","i944","i945",)
	x976 = NXArray(normal(size = (4,3,3,1,5,)), "i941","i945","i944","i942","i943",)
	x973 = x975 * x976
	assert set(x973.index_ids) == set([])
	assert x973.release_array().shape == ()
	assert x973.rank == 0
	x977 = NXArray(normal(size = (3,5,4,1,)), "i946","i947","i932","i934",)
	x978 = NXArray(normal(size = (1,1,3,2,5,)), "i931","i933","i946","i930","i947",)
	x974 = x977 * x978
	assert set(x974.index_ids) == set(["i932","i934","i933","i931","i930",])
	assert x974.release_array("i932","i934","i933","i931","i930",).shape == (4,1,1,1,2,)
	assert x974.rank == 5
	x966 = x973 * x974
	assert set(x966.index_ids) == set(["i932","i934","i933","i931","i930",])
	assert x966.release_array("i932","i934","i933","i931","i930",).shape == (4,1,1,1,2,)
	assert x966.rank == 5
	x963 = x965 * x966
	assert set(x963.index_ids) == set(["i934","i931","i932","i933","i930",])
	assert x963.release_array("i934","i931","i932","i933","i930",).shape == (1,1,4,1,2,)
	assert x963.rank == 5
	x983 = NXArray(normal(size = (5,)), "i953",)
	x984 = NXArray(normal(size = (4,5,1,1,5,)), "i949","i953","i952","i948","i951",)
	x981 = x983 * x984
	assert set(x981.index_ids) == set(["i949","i948","i951","i952",])
	assert x981.release_array("i949","i948","i951","i952",).shape == (4,1,5,1,)
	assert x981.rank == 4
	x985 = NXArray(normal(size = (2,5,4,1,2,)), "i954","i951","i955","i952","i872",)
	x986 = NXArray(normal(size = (1,2,5,4,)), "i934","i954","i950","i955",)
	x982 = x985 * x986
	assert set(x982.index_ids) == set(["i951","i952","i872","i950","i934",])
	assert x982.release_array("i951","i952","i872","i950","i934",).shape == (5,1,2,5,1,)
	assert x982.rank == 5
	x979 = x981 * x982
	assert set(x979.index_ids) == set(["i949","i948","i950","i872","i934",])
	assert x979.release_array("i949","i948","i950","i872","i934",).shape == (4,1,5,2,1,)
	assert x979.rank == 5
	x989 = NXArray(normal(size = (5,5,4,1,3,)), "i950","i957","i949","i948","i956",)
	x990 = NXArray(normal(size = ()), )
	x987 = x989 * x990
	assert set(x987.index_ids) == set(["i950","i956","i948","i957","i949",])
	assert x987.release_array("i950","i956","i948","i957","i949",).shape == (5,3,1,5,4,)
	assert x987.rank == 5
	x991 = NXArray(normal(size = (4,3,1,3,3,)), "i960","i958","i961","i959","i956",)
	x992 = NXArray(normal(size = (3,1,3,4,5,)), "i959","i961","i958","i960","i957",)
	x988 = x991 * x992
	assert set(x988.index_ids) == set(["i956","i957",])
	assert x988.release_array("i956","i957",).shape == (3,5,)
	assert x988.rank == 2
	x980 = x987 * x988
	assert set(x980.index_ids) == set(["i950","i949","i948",])
	assert x980.release_array("i950","i949","i948",).shape == (5,4,1,)
	assert x980.rank == 3
	x964 = x979 * x980
	assert set(x964.index_ids) == set(["i872","i934",])
	assert x964.release_array("i872","i934",).shape == (2,1,)
	assert x964.rank == 2
	x961 = x963 * x964
	assert set(x961.index_ids) == set(["i931","i930","i933","i932","i872",])
	assert x961.release_array("i931","i930","i933","i932","i872",).shape == (1,2,1,4,2,)
	assert x961.rank == 5
	x999 = NXArray(normal(size = (3,5,4,4,3,)), "i966","i967","i964","i962","i963",)
	x1000 = NXArray(normal(size = (3,5,)), "i965","i967",)
	x997 = x999 * x1000
	assert set(x997.index_ids) == set(["i964","i962","i966","i963","i965",])
	assert x997.release_array("i964","i962","i966","i963","i965",).shape == (4,4,3,3,3,)
	assert x997.rank == 5
	x1001 = NXArray(normal(size = (2,5,2,1,2,)), "i970","i968","i971","i969","i972",)
	x1002 = NXArray(normal(size = (2,1,2,5,2,)), "i971","i969","i970","i968","i972",)
	x998 = x1001 * x1002
	assert set(x998.index_ids) == set([])
	assert x998.release_array().shape == ()
	assert x998.rank == 0
	x995 = x997 * x998
	assert set(x995.index_ids) == set(["i964","i962","i963","i965","i966",])
	assert x995.release_array("i964","i962","i963","i965","i966",).shape == (4,4,3,3,3,)
	assert x995.rank == 5
	x1005 = NXArray(normal(size = (1,2,2,1,)), "i977","i974","i976","i975",)
	x1006 = NXArray(normal(size = (1,1,2,2,2,)), "i977","i975","i974","i976","i973",)
	x1003 = x1005 * x1006
	assert set(x1003.index_ids) == set(["i973",])
	assert x1003.release_array("i973",).shape == (2,)
	assert x1003.rank == 1
	x1007 = NXArray(normal(size = (3,3,5,2,3,)), "i965","i966","i978","i973","i963",)
	x1008 = NXArray(normal(size = (5,4,)), "i978","i964",)
	x1004 = x1007 * x1008
	assert set(x1004.index_ids) == set(["i966","i973","i963","i965","i964",])
	assert x1004.release_array("i966","i973","i963","i965","i964",).shape == (3,2,3,3,4,)
	assert x1004.rank == 5
	x996 = x1003 * x1004
	assert set(x996.index_ids) == set(["i963","i965","i966","i964",])
	assert x996.release_array("i963","i965","i966","i964",).shape == (3,3,3,4,)
	assert x996.rank == 4
	x993 = x995 * x996
	assert set(x993.index_ids) == set(["i962",])
	assert x993.release_array("i962",).shape == (4,)
	assert x993.rank == 1
	x1013 = NXArray(normal(size = (5,3,2,5,4,)), "i985","i982","i981","i983","i984",)
	x1014 = NXArray(normal(size = (2,5,5,3,4,)), "i981","i985","i983","i982","i984",)
	x1011 = x1013 * x1014
	assert set(x1011.index_ids) == set([])
	assert x1011.release_array().shape == ()
	assert x1011.rank == 0
	x1015 = NXArray(normal(size = ()), )
	x1016 = NXArray(normal(size = (4,5,1,2,4,)), "i962","i979","i931","i930","i980",)
	x1012 = x1015 * x1016
	assert set(x1012.index_ids) == set(["i931","i980","i962","i979","i930",])
	assert x1012.release_array("i931","i980","i962","i979","i930",).shape == (1,4,4,5,2,)
	assert x1012.rank == 5
	x1009 = x1011 * x1012
	assert set(x1009.index_ids) == set(["i931","i962","i979","i980","i930",])
	assert x1009.release_array("i931","i962","i979","i980","i930",).shape == (1,4,5,4,2,)
	assert x1009.rank == 5
	x1019 = NXArray(normal(size = (1,1,1,5,)), "i987","i933","i986","i988",)
	x1020 = NXArray(normal(size = (5,5,1,4,4,)), "i988","i979","i987","i980","i932",)
	x1017 = x1019 * x1020
	assert set(x1017.index_ids) == set(["i933","i986","i979","i932","i980",])
	assert x1017.release_array("i933","i986","i979","i932","i980",).shape == (1,1,5,4,4,)
	assert x1017.rank == 5
	x1021 = NXArray(normal(size = (1,2,3,4,)), "i992","i991","i989","i990",)
	x1022 = NXArray(normal(size = (1,1,4,2,3,)), "i986","i992","i990","i991","i989",)
	x1018 = x1021 * x1022
	assert set(x1018.index_ids) == set(["i986",])
	assert x1018.release_array("i986",).shape == (1,)
	assert x1018.rank == 1
	x1010 = x1017 * x1018
	assert set(x1010.index_ids) == set(["i933","i980","i979","i932",])
	assert x1010.release_array("i933","i980","i979","i932",).shape == (1,4,5,4,)
	assert x1010.rank == 4
	x994 = x1009 * x1010
	assert set(x994.index_ids) == set(["i931","i930","i962","i933","i932",])
	assert x994.release_array("i931","i930","i962","i933","i932",).shape == (1,2,4,1,4,)
	assert x994.rank == 5
	x962 = x993 * x994
	assert set(x962.index_ids) == set(["i931","i930","i933","i932",])
	assert x962.release_array("i931","i930","i933","i932",).shape == (1,2,1,4,)
	assert x962.rank == 4
	x898 = x961 * x962
	assert set(x898.index_ids) == set(["i872",])
	assert x898.release_array("i872",).shape == (2,)
	assert x898.rank == 1
	x770 = x897 * x898
	assert set(x770.index_ids) == set(["i754","i755","i756","i503",])
	assert x770.release_array("i754","i755","i756","i503",).shape == (2,4,1,5,)
	assert x770.rank == 4
	x514 = x769 * x770
	assert set(x514.index_ids) == set(["i3","i504","i503",])
	assert x514.release_array("i3","i504","i503",).shape == (5,4,5,)
	assert x514.rank == 3
	x2 = x513 * x514
	assert set(x2.index_ids) == set(["i4","i6","i5","i3",])
	assert x2.release_array("i4","i6","i5","i3",).shape == (3,4,3,5,)
	assert x2.rank == 4
	x0 = x1 * x2
	assert set(x0.index_ids) == set(["i0","i1","i2","i3","i4",])
	assert x0.release_array("i0","i1","i2","i3","i4",).shape == (3,2,3,5,3,)
	assert x0.rank == 5

# ----------------------------------------------------------------------------------- #

	x1025 = x17
	x1026 = x18
	x1027 = x1025 * x1026
	assert set(x1027.index_ids) == set(["i18","i17","i16","i13",])
	assert x1027.release_array("i18","i17","i16","i13",).shape == (4,1,2,3,)
	assert x1027.rank == 4
	x1023 = x19
	x1029 = x1027 * x1023
	assert set(x1029.index_ids) == set(["i17","i13","i21","i15","i14",])
	assert x1029.release_array("i17","i13","i21","i15","i14",).shape == (1,3,1,5,3,)
	assert x1029.rank == 5
	x1024 = x20
	x1028 = x1029 * x1024
	assert set(x1028.index_ids) == set(["i13","i15","i14",])
	assert x1028.release_array("i13","i15","i14",).shape == (3,5,3,)
	assert x1028.rank == 3
	x1030 = x23
	x1031 = x24
	x1032 = x1030 * x1031
	assert set(x1032.index_ids) == set(["i12","i13","i22","i8","i14",])
	assert x1032.release_array("i12","i13","i22","i8","i14",).shape == (4,3,5,4,3,)
	assert x1032.rank == 5
	x1033 = x25
	x1034 = x26
	x1035 = x1033 * x1034
	assert set(x1035.index_ids) == set(["i15","i22",])
	assert x1035.release_array("i15","i22",).shape == (5,5,)
	assert x1035.rank == 2
	x1036 = x1032 * x1035
	assert set(x1036.index_ids) == set(["i12","i13","i8","i14","i15",])
	assert x1036.release_array("i12","i13","i8","i14","i15",).shape == (4,3,4,3,5,)
	assert x1036.rank == 5
	x1043 = x31
	x1044 = x32
	x1045 = x33
	x1046 = x34
	x1047 = x1045 * x1046
	assert set(x1047.index_ids) == set(["i26","i9","i29","i28","i27",])
	assert x1047.release_array("i26","i9","i29","i28","i27",).shape == (5,2,3,2,4,)
	assert x1047.rank == 5
	x1049 = x1044 * x1047
	assert set(x1049.index_ids) == set(["i30","i31","i26","i9",])
	assert x1049.release_array("i30","i31","i26","i9",).shape == (5,5,5,2,)
	assert x1049.rank == 4
	x1048 = x1043 * x1049
	assert set(x1048.index_ids) == set(["i26","i9",])
	assert x1048.release_array("i26","i9",).shape == (5,2,)
	assert x1048.rank == 2
	x1037 = x37
	x1038 = x38
	x1039 = x1037 * x1038
	assert set(x1039.index_ids) == set(["i10","i34","i11","i35",])
	assert x1039.release_array("i10","i34","i11","i35",).shape == (2,5,1,5,)
	assert x1039.rank == 4
	x1051 = x1048 * x1039
	assert set(x1051.index_ids) == set(["i26","i9","i10","i34","i11","i35",])
	assert x1051.release_array("i26","i9","i10","i34","i11","i35",).shape == (5,2,2,5,1,5,)
	assert x1051.rank == 6
	x1040 = x39
	x1041 = x40
	x1042 = x1040 * x1041
	assert set(x1042.index_ids) == set(["i12","i5","i26","i34","i35",])
	assert x1042.release_array("i12","i5","i26","i34","i35",).shape == (4,3,5,5,5,)
	assert x1042.rank == 5
	x1050 = x1051 * x1042
	assert set(x1050.index_ids) == set(["i9","i10","i11","i12","i5",])
	assert x1050.release_array("i9","i10","i11","i12","i5",).shape == (2,2,1,4,3,)
	assert x1050.rank == 5
	x1053 = x1036 * x1050
	assert set(x1053.index_ids) == set(["i13","i8","i14","i15","i9","i10","i11","i5",])
	assert x1053.release_array("i13","i8","i14","i15","i9","i10","i11","i5",).shape == (3,4,3,5,2,2,1,3,)
	assert x1053.rank == 8
	x1052 = x1028 * x1053
	assert set(x1052.index_ids) == set(["i8","i9","i10","i11","i5",])
	assert x1052.release_array("i8","i9","i10","i11","i5",).shape == (4,2,2,1,3,)
	assert x1052.rank == 5
	x1054 = x47
	x1055 = x48
	x1056 = x1054 * x1055
	assert set(x1056.index_ids) == set(["i42","i43","i40","i39","i41",])
	assert x1056.release_array("i42","i43","i40","i39","i41",).shape == (3,1,4,5,4,)
	assert x1056.rank == 5
	x1057 = x49
	x1058 = x50
	x1059 = x1057 * x1058
	assert set(x1059.index_ids) == set(["i40","i41","i39","i42","i43",])
	assert x1059.release_array("i40","i41","i39","i42","i43",).shape == (4,4,5,3,1,)
	assert x1059.rank == 5
	x1060 = x1056 * x1059
	assert set(x1060.index_ids) == set([])
	assert x1060.release_array().shape == ()
	assert x1060.rank == 0
	x1061 = x53
	x1062 = x54
	x1063 = x55
	x1064 = x56
	x1065 = x1063 * x1064
	assert set(x1065.index_ids) == set(["i46","i10","i45","i7",])
	assert x1065.release_array("i46","i10","i45","i7",).shape == (1,2,4,3,)
	assert x1065.rank == 4
	x1067 = x1062 * x1065
	assert set(x1067.index_ids) == set(["i47","i8","i48","i10","i7",])
	assert x1067.release_array("i47","i8","i48","i10","i7",).shape == (4,4,3,2,3,)
	assert x1067.rank == 5
	x1066 = x1061 * x1067
	assert set(x1066.index_ids) == set(["i9","i11","i8","i10","i7",])
	assert x1066.release_array("i9","i11","i8","i10","i7",).shape == (2,1,4,2,3,)
	assert x1066.rank == 5
	x1070 = x61
	x1071 = x62
	x1072 = x1070 * x1071
	assert set(x1072.index_ids) == set(["i50","i55",])
	assert x1072.release_array("i50","i55",).shape == (5,5,)
	assert x1072.rank == 2
	x1068 = x63
	x1074 = x1072 * x1068
	assert set(x1074.index_ids) == set(["i50","i59","i53","i51","i54",])
	assert x1074.release_array("i50","i59","i53","i51","i54",).shape == (5,1,5,1,5,)
	assert x1074.rank == 5
	x1069 = x64
	x1073 = x1074 * x1069
	assert set(x1073.index_ids) == set(["i50","i53","i51","i54","i52",])
	assert x1073.release_array("i50","i53","i51","i54","i52",).shape == (5,5,1,5,1,)
	assert x1073.rank == 5
	x1075 = x67
	x1076 = x68
	x1077 = x69
	x1078 = x70
	x1079 = x1077 * x1078
	assert set(x1079.index_ids) == set(["i60","i61","i54","i52","i51",])
	assert x1079.release_array("i60","i61","i54","i52","i51",).shape == (5,4,5,1,1,)
	assert x1079.rank == 5
	x1081 = x1076 * x1079
	assert set(x1081.index_ids) == set(["i62","i60","i61","i54","i52","i51",])
	assert x1081.release_array("i62","i60","i61","i54","i52","i51",).shape == (1,5,4,5,1,1,)
	assert x1081.rank == 6
	x1080 = x1075 * x1081
	assert set(x1080.index_ids) == set(["i50","i53","i54","i52","i51",])
	assert x1080.release_array("i50","i53","i54","i52","i51",).shape == (5,5,5,1,1,)
	assert x1080.rank == 5
	x1082 = x1073 * x1080
	assert set(x1082.index_ids) == set([])
	assert x1082.release_array().shape == ()
	assert x1082.rank == 0
	x1084 = x1066 * x1082
	assert set(x1084.index_ids) == set(["i9","i11","i8","i10","i7",])
	assert x1084.release_array("i9","i11","i8","i10","i7",).shape == (2,1,4,2,3,)
	assert x1084.rank == 5
	x1083 = x1060 * x1084
	assert set(x1083.index_ids) == set(["i9","i11","i8","i10","i7",])
	assert x1083.release_array("i9","i11","i8","i10","i7",).shape == (2,1,4,2,3,)
	assert x1083.rank == 5
	x1099 = x79
	x1100 = x80
	x1101 = x1099 * x1100
	assert set(x1101.index_ids) == set(["i2","i1","i70","i67","i69",])
	assert x1101.release_array("i2","i1","i70","i67","i69",).shape == (3,2,4,4,4,)
	assert x1101.rank == 5
	x1102 = x81
	x1103 = x82
	x1104 = x1102 * x1103
	assert set(x1104.index_ids) == set(["i70","i68","i69","i66",])
	assert x1104.release_array("i70","i68","i69","i66",).shape == (4,1,4,4,)
	assert x1104.rank == 4
	x1105 = x1101 * x1104
	assert set(x1105.index_ids) == set(["i2","i1","i67","i68","i66",])
	assert x1105.release_array("i2","i1","i67","i68","i66",).shape == (3,2,4,1,4,)
	assert x1105.rank == 5
	x1106 = x85
	x1107 = x86
	x1108 = x1106 * x1107
	assert set(x1108.index_ids) == set(["i72",])
	assert x1108.release_array("i72",).shape == (3,)
	assert x1108.rank == 1
	x1109 = x87
	x1110 = x88
	x1111 = x1109 * x1110
	assert set(x1111.index_ids) == set(["i67","i68","i72","i65","i7",])
	assert x1111.release_array("i67","i68","i72","i65","i7",).shape == (4,1,3,5,3,)
	assert x1111.rank == 5
	x1112 = x1108 * x1111
	assert set(x1112.index_ids) == set(["i67","i68","i65","i7",])
	assert x1112.release_array("i67","i68","i65","i7",).shape == (4,1,5,3,)
	assert x1112.rank == 4
	x1113 = x1105 * x1112
	assert set(x1113.index_ids) == set(["i2","i1","i66","i65","i7",])
	assert x1113.release_array("i2","i1","i66","i65","i7",).shape == (3,2,4,5,3,)
	assert x1113.rank == 5
	x1087 = x93
	x1088 = x94
	x1089 = x1087 * x1088
	assert set(x1089.index_ids) == set(["i82","i80","i79","i83","i81",])
	assert x1089.release_array("i82","i80","i79","i83","i81",).shape == (2,2,4,5,5,)
	assert x1089.rank == 5
	x1085 = x95
	x1091 = x1089 * x1085
	assert set(x1091.index_ids) == set(["i82","i80","i79","i83","i81","i87","i85","i84","i86","i88",])
	assert x1091.release_array("i82","i80","i79","i83","i81","i87","i85","i84","i86","i88",).shape == (2,2,4,5,5,2,5,2,2,4,)
	assert x1091.rank == 10
	x1086 = x96
	x1090 = x1091 * x1086
	assert set(x1090.index_ids) == set(["i82","i80","i79","i83","i81",])
	assert x1090.release_array("i82","i80","i79","i83","i81",).shape == (2,2,4,5,5,)
	assert x1090.rank == 5
	x1115 = x1113 * x1090
	assert set(x1115.index_ids) == set(["i2","i1","i66","i65","i7","i82","i80","i79","i83","i81",])
	assert x1115.release_array("i2","i1","i66","i65","i7","i82","i80","i79","i83","i81",).shape == (3,2,4,5,3,2,2,4,5,5,)
	assert x1115.rank == 10
	x1092 = x99
	x1093 = x100
	x1094 = x1092 * x1093
	assert set(x1094.index_ids) == set([])
	assert x1094.release_array().shape == ()
	assert x1094.rank == 0
	x1095 = x101
	x1096 = x102
	x1097 = x1095 * x1096
	assert set(x1097.index_ids) == set(["i81","i79","i80","i83","i82",])
	assert x1097.release_array("i81","i79","i80","i83","i82",).shape == (5,4,2,5,2,)
	assert x1097.rank == 5
	x1098 = x1094 * x1097
	assert set(x1098.index_ids) == set(["i81","i79","i80","i83","i82",])
	assert x1098.release_array("i81","i79","i80","i83","i82",).shape == (5,4,2,5,2,)
	assert x1098.rank == 5
	x1114 = x1115 * x1098
	assert set(x1114.index_ids) == set(["i2","i1","i66","i65","i7",])
	assert x1114.release_array("i2","i1","i66","i65","i7",).shape == (3,2,4,5,3,)
	assert x1114.rank == 5
	x1130 = x109
	x1131 = x110
	x1132 = x1130 * x1131
	assert set(x1132.index_ids) == set([])
	assert x1132.release_array().shape == ()
	assert x1132.rank == 0
	x1133 = x111
	x1134 = x112
	x1135 = x1133 * x1134
	assert set(x1135.index_ids) == set(["i97","i65","i98","i99","i96",])
	assert x1135.release_array("i97","i65","i98","i99","i96",).shape == (2,5,2,4,3,)
	assert x1135.rank == 5
	x1136 = x1132 * x1135
	assert set(x1136.index_ids) == set(["i97","i65","i98","i99","i96",])
	assert x1136.release_array("i97","i65","i98","i99","i96",).shape == (2,5,2,4,3,)
	assert x1136.rank == 5
	x1137 = x115
	x1138 = x116
	x1139 = x117
	x1140 = x118
	x1141 = x1139 * x1140
	assert set(x1141.index_ids) == set(["i98","i106","i105","i108","i107",])
	assert x1141.release_array("i98","i106","i105","i108","i107",).shape == (2,3,2,4,3,)
	assert x1141.rank == 5
	x1143 = x1138 * x1141
	assert set(x1143.index_ids) == set(["i109","i110","i99","i98","i106","i105","i107",])
	assert x1143.release_array("i109","i110","i99","i98","i106","i105","i107",).shape == (1,4,4,2,3,2,3,)
	assert x1143.rank == 7
	x1142 = x1137 * x1143
	assert set(x1142.index_ids) == set(["i99","i98",])
	assert x1142.release_array("i99","i98",).shape == (4,2,)
	assert x1142.rank == 2
	x1144 = x1136 * x1142
	assert set(x1144.index_ids) == set(["i97","i65","i96",])
	assert x1144.release_array("i97","i65","i96",).shape == (2,5,3,)
	assert x1144.rank == 3
	x1118 = x123
	x1119 = x124
	x1120 = x1118 * x1119
	assert set(x1120.index_ids) == set(["i113","i114","i115","i117","i116",])
	assert x1120.release_array("i113","i114","i115","i117","i116",).shape == (5,4,3,2,4,)
	assert x1120.rank == 5
	x1116 = x125
	x1122 = x1120 * x1116
	assert set(x1122.index_ids) == set(["i115","i118",])
	assert x1122.release_array("i115","i118",).shape == (3,2,)
	assert x1122.rank == 2
	x1117 = x126
	x1121 = x1122 * x1117
	assert set(x1121.index_ids) == set([])
	assert x1121.release_array().shape == ()
	assert x1121.rank == 0
	x1146 = x1144 * x1121
	assert set(x1146.index_ids) == set(["i97","i65","i96",])
	assert x1146.release_array("i97","i65","i96",).shape == (2,5,3,)
	assert x1146.rank == 3
	x1123 = x129
	x1124 = x130
	x1125 = x131
	x1126 = x132
	x1127 = x1125 * x1126
	assert set(x1127.index_ids) == set(["i120","i97","i66","i0","i119",])
	assert x1127.release_array("i120","i97","i66","i0","i119",).shape == (2,2,4,3,3,)
	assert x1127.rank == 5
	x1129 = x1124 * x1127
	assert set(x1129.index_ids) == set(["i121","i122","i120","i97","i66","i0",])
	assert x1129.release_array("i121","i122","i120","i97","i66","i0",).shape == (5,2,2,2,4,3,)
	assert x1129.rank == 6
	x1128 = x1123 * x1129
	assert set(x1128.index_ids) == set(["i96","i6","i97","i66","i0",])
	assert x1128.release_array("i96","i6","i97","i66","i0",).shape == (3,4,2,4,3,)
	assert x1128.rank == 5
	x1145 = x1146 * x1128
	assert set(x1145.index_ids) == set(["i65","i6","i66","i0",])
	assert x1145.release_array("i65","i6","i66","i0",).shape == (5,4,4,3,)
	assert x1145.rank == 4
	x1147 = x1114 * x1145
	assert set(x1147.index_ids) == set(["i2","i1","i7","i6","i0",])
	assert x1147.release_array("i2","i1","i7","i6","i0",).shape == (3,2,3,4,3,)
	assert x1147.rank == 5
	x1149 = x1083 * x1147
	assert set(x1149.index_ids) == set(["i9","i11","i8","i10","i2","i1","i6","i0",])
	assert x1149.release_array("i9","i11","i8","i10","i2","i1","i6","i0",).shape == (2,1,4,2,3,2,4,3,)
	assert x1149.rank == 8
	x1148 = x1052 * x1149
	assert set(x1148.index_ids) == set(["i5","i2","i1","i6","i0",])
	assert x1148.release_array("i5","i2","i1","i6","i0",).shape == (3,3,2,4,3,)
	assert x1148.rank == 5
	x1164 = x143
	x1165 = x144
	x1166 = x1164 * x1165
	assert set(x1166.index_ids) == set(["i134","i133","i135","i132","i136",])
	assert x1166.release_array("i134","i133","i135","i132","i136",).shape == (2,2,5,5,5,)
	assert x1166.rank == 5
	x1167 = x145
	x1168 = x146
	x1169 = x1167 * x1168
	assert set(x1169.index_ids) == set([])
	assert x1169.release_array().shape == ()
	assert x1169.rank == 0
	x1170 = x149
	x1171 = x150
	x1172 = x1170 * x1171
	assert set(x1172.index_ids) == set(["i143","i135","i134","i144","i136",])
	assert x1172.release_array("i143","i135","i134","i144","i136",).shape == (4,5,2,4,5,)
	assert x1172.rank == 5
	x1173 = x151
	x1174 = x152
	x1175 = x1173 * x1174
	assert set(x1175.index_ids) == set(["i130","i143","i144",])
	assert x1175.release_array("i130","i143","i144",).shape == (4,4,4,)
	assert x1175.rank == 3
	x1176 = x1172 * x1175
	assert set(x1176.index_ids) == set(["i135","i134","i136","i130",])
	assert x1176.release_array("i135","i134","i136","i130",).shape == (5,2,5,4,)
	assert x1176.rank == 4
	x1178 = x1169 * x1176
	assert set(x1178.index_ids) == set(["i135","i134","i136","i130",])
	assert x1178.release_array("i135","i134","i136","i130",).shape == (5,2,5,4,)
	assert x1178.rank == 4
	x1177 = x1166 * x1178
	assert set(x1177.index_ids) == set(["i133","i132","i130",])
	assert x1177.release_array("i133","i132","i130",).shape == (2,5,4,)
	assert x1177.rank == 3
	x1150 = x157
	x1151 = x158
	x1152 = x159
	x1153 = x160
	x1154 = x1152 * x1153
	assert set(x1154.index_ids) == set(["i149","i132","i128","i127","i148",])
	assert x1154.release_array("i149","i132","i128","i127","i148",).shape == (1,5,2,3,2,)
	assert x1154.rank == 5
	x1156 = x1151 * x1154
	assert set(x1156.index_ids) == set(["i154","i150","i153","i151","i152","i149","i132","i128","i127","i148",])
	assert x1156.release_array("i154","i150","i153","i151","i152","i149","i132","i128","i127","i148",).shape == (3,5,2,3,3,1,5,2,3,2,)
	assert x1156.rank == 10
	x1155 = x1150 * x1156
	assert set(x1155.index_ids) == set(["i149","i132","i128","i127","i148",])
	assert x1155.release_array("i149","i132","i128","i127","i148",).shape == (1,5,2,3,2,)
	assert x1155.rank == 5
	x1180 = x1177 * x1155
	assert set(x1180.index_ids) == set(["i133","i130","i149","i128","i127","i148",])
	assert x1180.release_array("i133","i130","i149","i128","i127","i148",).shape == (2,4,1,2,3,2,)
	assert x1180.rank == 6
	x1157 = x163
	x1158 = x164
	x1159 = x1157 * x1158
	assert set(x1159.index_ids) == set(["i149","i148","i156","i155","i157",])
	assert x1159.release_array("i149","i148","i156","i155","i157",).shape == (1,2,4,1,2,)
	assert x1159.rank == 5
	x1160 = x165
	x1161 = x166
	x1162 = x1160 * x1161
	assert set(x1162.index_ids) == set(["i157","i131","i155","i156","i133",])
	assert x1162.release_array("i157","i131","i155","i156","i133",).shape == (2,2,1,4,2,)
	assert x1162.rank == 5
	x1163 = x1159 * x1162
	assert set(x1163.index_ids) == set(["i149","i148","i131","i133",])
	assert x1163.release_array("i149","i148","i131","i133",).shape == (1,2,2,2,)
	assert x1163.rank == 4
	x1179 = x1180 * x1163
	assert set(x1179.index_ids) == set(["i130","i128","i127","i131",])
	assert x1179.release_array("i130","i128","i127","i131",).shape == (4,2,3,2,)
	assert x1179.rank == 4
	x1187 = x173
	x1188 = x174
	x1189 = x175
	x1190 = x176
	x1191 = x1189 * x1190
	assert set(x1191.index_ids) == set(["i164","i165",])
	assert x1191.release_array("i164","i165",).shape == (4,1,)
	assert x1191.rank == 2
	x1193 = x1188 * x1191
	assert set(x1193.index_ids) == set(["i167","i166","i162","i164",])
	assert x1193.release_array("i167","i166","i162","i164",).shape == (3,3,3,4,)
	assert x1193.rank == 4
	x1192 = x1187 * x1193
	assert set(x1192.index_ids) == set(["i163","i161","i162",])
	assert x1192.release_array("i163","i161","i162",).shape == (2,2,3,)
	assert x1192.rank == 3
	x1181 = x179
	x1182 = x180
	x1183 = x1181 * x1182
	assert set(x1183.index_ids) == set(["i172","i163","i161","i162","i126",])
	assert x1183.release_array("i172","i163","i161","i162","i126",).shape == (4,2,2,3,4,)
	assert x1183.rank == 5
	x1195 = x1192 * x1183
	assert set(x1195.index_ids) == set(["i172","i126",])
	assert x1195.release_array("i172","i126",).shape == (4,4,)
	assert x1195.rank == 2
	x1184 = x181
	x1185 = x182
	x1186 = x1184 * x1185
	assert set(x1186.index_ids) == set(["i160","i172",])
	assert x1186.release_array("i160","i172",).shape == (5,4,)
	assert x1186.rank == 2
	x1194 = x1195 * x1186
	assert set(x1194.index_ids) == set(["i126","i160",])
	assert x1194.release_array("i126","i160",).shape == (4,5,)
	assert x1194.rank == 2
	x1204 = x187
	x1205 = x188
	x1206 = x1204 * x1205
	assert set(x1206.index_ids) == set(["i130","i177",])
	assert x1206.release_array("i130","i177",).shape == (4,3,)
	assert x1206.rank == 2
	x1202 = x189
	x1208 = x1206 * x1202
	assert set(x1208.index_ids) == set(["i130","i181","i182","i176","i160",])
	assert x1208.release_array("i130","i181","i182","i176","i160",).shape == (4,4,2,3,5,)
	assert x1208.rank == 5
	x1203 = x190
	x1207 = x1208 * x1203
	assert set(x1207.index_ids) == set(["i130","i176","i160","i131","i129",])
	assert x1207.release_array("i130","i176","i160","i131","i129",).shape == (4,3,5,2,3,)
	assert x1207.rank == 5
	x1196 = x193
	x1197 = x194
	x1198 = x1196 * x1197
	assert set(x1198.index_ids) == set(["i184","i183","i185",])
	assert x1198.release_array("i184","i183","i185",).shape == (1,5,1,)
	assert x1198.rank == 3
	x1210 = x1207 * x1198
	assert set(x1210.index_ids) == set(["i130","i176","i160","i131","i129","i184","i183","i185",])
	assert x1210.release_array("i130","i176","i160","i131","i129","i184","i183","i185",).shape == (4,3,5,2,3,1,5,1,)
	assert x1210.rank == 8
	x1199 = x195
	x1200 = x196
	x1201 = x1199 * x1200
	assert set(x1201.index_ids) == set(["i185","i125","i183","i176","i184",])
	assert x1201.release_array("i185","i125","i183","i176","i184",).shape == (1,4,5,3,1,)
	assert x1201.rank == 5
	x1209 = x1210 * x1201
	assert set(x1209.index_ids) == set(["i130","i160","i131","i129","i125",])
	assert x1209.release_array("i130","i160","i131","i129","i125",).shape == (4,5,2,3,4,)
	assert x1209.rank == 5
	x1211 = x1194 * x1209
	assert set(x1211.index_ids) == set(["i126","i130","i131","i129","i125",])
	assert x1211.release_array("i126","i130","i131","i129","i125",).shape == (4,4,2,3,4,)
	assert x1211.rank == 5
	x1226 = x205
	x1227 = x206
	x1228 = x1226 * x1227
	assert set(x1228.index_ids) == set(["i190","i193",])
	assert x1228.release_array("i190","i193",).shape == (1,5,)
	assert x1228.rank == 2
	x1229 = x207
	x1230 = x208
	x1231 = x1229 * x1230
	assert set(x1231.index_ids) == set(["i192","i191","i193","i128","i189",])
	assert x1231.release_array("i192","i191","i193","i128","i189",).shape == (1,1,5,2,5,)
	assert x1231.rank == 5
	x1234 = x211
	x1235 = x212
	x1236 = x1234 * x1235
	assert set(x1236.index_ids) == set(["i199","i201","i198","i202","i200",])
	assert x1236.release_array("i199","i201","i198","i202","i200",).shape == (5,3,1,5,1,)
	assert x1236.rank == 5
	x1232 = x213
	x1238 = x1236 * x1232
	assert set(x1238.index_ids) == set([])
	assert x1238.release_array().shape == ()
	assert x1238.rank == 0
	x1233 = x214
	x1237 = x1238 * x1233
	assert set(x1237.index_ids) == set([])
	assert x1237.release_array().shape == ()
	assert x1237.rank == 0
	x1240 = x1231 * x1237
	assert set(x1240.index_ids) == set(["i192","i191","i193","i128","i189",])
	assert x1240.release_array("i192","i191","i193","i128","i189",).shape == (1,1,5,2,5,)
	assert x1240.rank == 5
	x1239 = x1228 * x1240
	assert set(x1239.index_ids) == set(["i190","i192","i191","i128","i189",])
	assert x1239.release_array("i190","i192","i191","i128","i189",).shape == (1,1,1,2,5,)
	assert x1239.rank == 5
	x1214 = x219
	x1215 = x220
	x1216 = x1214 * x1215
	assert set(x1216.index_ids) == set([])
	assert x1216.release_array().shape == ()
	assert x1216.rank == 0
	x1212 = x221
	x1218 = x1216 * x1212
	assert set(x1218.index_ids) == set(["i212","i191","i213","i206","i204",])
	assert x1218.release_array("i212","i191","i213","i206","i204",).shape == (3,1,5,1,4,)
	assert x1218.rank == 5
	x1213 = x222
	x1217 = x1218 * x1213
	assert set(x1217.index_ids) == set(["i191","i206","i204","i190","i205",])
	assert x1217.release_array("i191","i206","i204","i190","i205",).shape == (1,1,4,1,3,)
	assert x1217.rank == 5
	x1242 = x1239 * x1217
	assert set(x1242.index_ids) == set(["i192","i128","i189","i206","i204","i205",])
	assert x1242.release_array("i192","i128","i189","i206","i204","i205",).shape == (1,2,5,1,4,3,)
	assert x1242.rank == 6
	x1219 = x225
	x1220 = x226
	x1221 = x227
	x1222 = x228
	x1223 = x1221 * x1222
	assert set(x1223.index_ids) == set(["i215","i214","i205",])
	assert x1223.release_array("i215","i214","i205",).shape == (5,2,3,)
	assert x1223.rank == 3
	x1225 = x1220 * x1223
	assert set(x1225.index_ids) == set(["i216","i217","i205",])
	assert x1225.release_array("i216","i217","i205",).shape == (4,5,3,)
	assert x1225.rank == 3
	x1224 = x1219 * x1225
	assert set(x1224.index_ids) == set(["i192","i204","i206","i205",])
	assert x1224.release_array("i192","i204","i206","i205",).shape == (1,4,1,3,)
	assert x1224.rank == 4
	x1241 = x1242 * x1224
	assert set(x1241.index_ids) == set(["i128","i189",])
	assert x1241.release_array("i128","i189",).shape == (2,5,)
	assert x1241.rank == 2
	x1251 = x235
	x1252 = x236
	x1253 = x1251 * x1252
	assert set(x1253.index_ids) == set(["i223","i227","i226","i225","i224",])
	assert x1253.release_array("i223","i227","i226","i225","i224",).shape == (5,5,3,1,5,)
	assert x1253.rank == 5
	x1249 = x237
	x1255 = x1253 * x1249
	assert set(x1255.index_ids) == set([])
	assert x1255.release_array().shape == ()
	assert x1255.rank == 0
	x1250 = x238
	x1254 = x1255 * x1250
	assert set(x1254.index_ids) == set([])
	assert x1254.release_array().shape == ()
	assert x1254.rank == 0
	x1243 = x241
	x1244 = x242
	x1245 = x1243 * x1244
	assert set(x1245.index_ids) == set(["i222","i129","i221","i125","i126",])
	assert x1245.release_array("i222","i129","i221","i125","i126",).shape == (3,3,2,4,4,)
	assert x1245.rank == 5
	x1257 = x1254 * x1245
	assert set(x1257.index_ids) == set(["i222","i129","i221","i125","i126",])
	assert x1257.release_array("i222","i129","i221","i125","i126",).shape == (3,3,2,4,4,)
	assert x1257.rank == 5
	x1246 = x243
	x1247 = x244
	x1248 = x1246 * x1247
	assert set(x1248.index_ids) == set([])
	assert x1248.release_array().shape == ()
	assert x1248.rank == 0
	x1256 = x1257 * x1248
	assert set(x1256.index_ids) == set(["i222","i129","i221","i125","i126",])
	assert x1256.release_array("i222","i129","i221","i125","i126",).shape == (3,3,2,4,4,)
	assert x1256.rank == 5
	x1264 = x249
	x1265 = x250
	x1266 = x251
	x1267 = x252
	x1268 = x1266 * x1267
	assert set(x1268.index_ids) == set(["i235","i236","i239","i237","i238",])
	assert x1268.release_array("i235","i236","i239","i237","i238",).shape == (3,1,2,5,1,)
	assert x1268.rank == 5
	x1270 = x1265 * x1268
	assert set(x1270.index_ids) == set(["i240","i241","i235","i236","i239","i237",])
	assert x1270.release_array("i240","i241","i235","i236","i239","i237",).shape == (1,4,3,1,2,5,)
	assert x1270.rank == 6
	x1269 = x1264 * x1270
	assert set(x1269.index_ids) == set(["i235",])
	assert x1269.release_array("i235",).shape == (3,)
	assert x1269.rank == 1
	x1258 = x255
	x1259 = x256
	x1260 = x1258 * x1259
	assert set(x1260.index_ids) == set(["i127","i222","i235","i244","i221",])
	assert x1260.release_array("i127","i222","i235","i244","i221",).shape == (3,3,3,1,2,)
	assert x1260.rank == 5
	x1272 = x1269 * x1260
	assert set(x1272.index_ids) == set(["i127","i222","i244","i221",])
	assert x1272.release_array("i127","i222","i244","i221",).shape == (3,3,1,2,)
	assert x1272.rank == 4
	x1261 = x257
	x1262 = x258
	x1263 = x1261 * x1262
	assert set(x1263.index_ids) == set(["i244","i189",])
	assert x1263.release_array("i244","i189",).shape == (1,5,)
	assert x1263.rank == 2
	x1271 = x1272 * x1263
	assert set(x1271.index_ids) == set(["i127","i222","i221","i189",])
	assert x1271.release_array("i127","i222","i221","i189",).shape == (3,3,2,5,)
	assert x1271.rank == 4
	x1273 = x1256 * x1271
	assert set(x1273.index_ids) == set(["i129","i125","i126","i127","i189",])
	assert x1273.release_array("i129","i125","i126","i127","i189",).shape == (3,4,4,3,5,)
	assert x1273.rank == 5
	x1274 = x1241 * x1273
	assert set(x1274.index_ids) == set(["i128","i129","i125","i126","i127",])
	assert x1274.release_array("i128","i129","i125","i126","i127",).shape == (2,3,4,4,3,)
	assert x1274.rank == 5
	x1276 = x1211 * x1274
	assert set(x1276.index_ids) == set(["i130","i131","i128","i127",])
	assert x1276.release_array("i130","i131","i128","i127",).shape == (4,2,2,3,)
	assert x1276.rank == 4
	x1275 = x1179 * x1276
	assert set(x1275.index_ids) == set([])
	assert x1275.release_array().shape == ()
	assert x1275.rank == 0
	x1479 = x271
	x1480 = x272
	x1481 = x1479 * x1480
	assert set(x1481.index_ids) == set(["i260","i264","i261","i262","i263",])
	assert x1481.release_array("i260","i264","i261","i262","i263",).shape == (5,5,2,1,5,)
	assert x1481.rank == 5
	x1482 = x273
	x1483 = x274
	x1484 = x1482 * x1483
	assert set(x1484.index_ids) == set(["i263","i264",])
	assert x1484.release_array("i263","i264",).shape == (5,5,)
	assert x1484.rank == 2
	x1485 = x1481 * x1484
	assert set(x1485.index_ids) == set(["i260","i261","i262",])
	assert x1485.release_array("i260","i261","i262",).shape == (5,2,1,)
	assert x1485.rank == 3
	x1486 = x277
	x1487 = x278
	x1488 = x1486 * x1487
	assert set(x1488.index_ids) == set([])
	assert x1488.release_array().shape == ()
	assert x1488.rank == 0
	x1489 = x279
	x1490 = x280
	x1491 = x1489 * x1490
	assert set(x1491.index_ids) == set(["i261","i259","i262","i258","i260",])
	assert x1491.release_array("i261","i259","i262","i258","i260",).shape == (2,1,1,5,5,)
	assert x1491.rank == 5
	x1492 = x1488 * x1491
	assert set(x1492.index_ids) == set(["i261","i259","i262","i258","i260",])
	assert x1492.release_array("i261","i259","i262","i258","i260",).shape == (2,1,1,5,5,)
	assert x1492.rank == 5
	x1493 = x1485 * x1492
	assert set(x1493.index_ids) == set(["i259","i258",])
	assert x1493.release_array("i259","i258",).shape == (1,5,)
	assert x1493.rank == 2
	x1467 = x285
	x1468 = x286
	x1469 = x1467 * x1468
	assert set(x1469.index_ids) == set(["i276","i275","i274","i273","i255",])
	assert x1469.release_array("i276","i275","i274","i273","i255",).shape == (4,3,2,1,1,)
	assert x1469.rank == 5
	x1465 = x287
	x1471 = x1469 * x1465
	assert set(x1471.index_ids) == set(["i273","i255","i278","i279",])
	assert x1471.release_array("i273","i255","i278","i279",).shape == (1,1,2,2,)
	assert x1471.rank == 4
	x1466 = x288
	x1470 = x1471 * x1466
	assert set(x1470.index_ids) == set(["i273","i255",])
	assert x1470.release_array("i273","i255",).shape == (1,1,)
	assert x1470.rank == 2
	x1495 = x1493 * x1470
	assert set(x1495.index_ids) == set(["i259","i258","i273","i255",])
	assert x1495.release_array("i259","i258","i273","i255",).shape == (1,5,1,1,)
	assert x1495.rank == 4
	x1474 = x291
	x1475 = x292
	x1476 = x1474 * x1475
	assert set(x1476.index_ids) == set(["i280","i256","i281","i257","i273",])
	assert x1476.release_array("i280","i256","i281","i257","i273",).shape == (2,3,3,5,1,)
	assert x1476.rank == 5
	x1472 = x293
	x1478 = x1476 * x1472
	assert set(x1478.index_ids) == set(["i256","i257","i273","i259","i258","i283",])
	assert x1478.release_array("i256","i257","i273","i259","i258","i283",).shape == (3,5,1,1,5,1,)
	assert x1478.rank == 6
	x1473 = x294
	x1477 = x1478 * x1473
	assert set(x1477.index_ids) == set(["i256","i257","i273","i259","i258",])
	assert x1477.release_array("i256","i257","i273","i259","i258",).shape == (3,5,1,1,5,)
	assert x1477.rank == 5
	x1494 = x1495 * x1477
	assert set(x1494.index_ids) == set(["i255","i256","i257",])
	assert x1494.release_array("i255","i256","i257",).shape == (1,3,5,)
	assert x1494.rank == 3
	x1510 = x301
	x1511 = x302
	x1512 = x1510 * x1511
	assert set(x1512.index_ids) == set(["i285","i284","i256","i250",])
	assert x1512.release_array("i285","i284","i256","i250",).shape == (3,2,3,3,)
	assert x1512.rank == 4
	x1513 = x303
	x1514 = x304
	x1515 = x1513 * x1514
	assert set(x1515.index_ids) == set(["i255","i284","i254","i257","i285",])
	assert x1515.release_array("i255","i284","i254","i257","i285",).shape == (1,2,1,5,3,)
	assert x1515.rank == 5
	x1516 = x307
	x1517 = x308
	x1518 = x309
	x1519 = x310
	x1520 = x1518 * x1519
	assert set(x1520.index_ids) == set(["i289","i292","i293","i290","i291",])
	assert x1520.release_array("i289","i292","i293","i290","i291",).shape == (5,3,2,1,3,)
	assert x1520.rank == 5
	x1522 = x1517 * x1520
	assert set(x1522.index_ids) == set([])
	assert x1522.release_array().shape == ()
	assert x1522.rank == 0
	x1521 = x1516 * x1522
	assert set(x1521.index_ids) == set([])
	assert x1521.release_array().shape == ()
	assert x1521.rank == 0
	x1524 = x1515 * x1521
	assert set(x1524.index_ids) == set(["i255","i284","i254","i257","i285",])
	assert x1524.release_array("i255","i284","i254","i257","i285",).shape == (1,2,1,5,3,)
	assert x1524.rank == 5
	x1523 = x1512 * x1524
	assert set(x1523.index_ids) == set(["i256","i250","i255","i254","i257",])
	assert x1523.release_array("i256","i250","i255","i254","i257",).shape == (3,3,1,1,5,)
	assert x1523.rank == 5
	x1496 = x315
	x1497 = x316
	x1498 = x317
	x1499 = x318
	x1500 = x1498 * x1499
	assert set(x1500.index_ids) == set(["i301","i300","i296","i299",])
	assert x1500.release_array("i301","i300","i296","i299",).shape == (3,2,1,2,)
	assert x1500.rank == 4
	x1502 = x1497 * x1500
	assert set(x1502.index_ids) == set(["i298","i302","i295","i303","i301","i300","i296","i299",])
	assert x1502.release_array("i298","i302","i295","i303","i301","i300","i296","i299",).shape == (5,5,4,4,3,2,1,2,)
	assert x1502.rank == 8
	x1501 = x1496 * x1502
	assert set(x1501.index_ids) == set(["i297","i298","i295","i296","i299",])
	assert x1501.release_array("i297","i298","i295","i296","i299",).shape == (1,5,4,1,2,)
	assert x1501.rank == 5
	x1526 = x1523 * x1501
	assert set(x1526.index_ids) == set(["i256","i250","i255","i254","i257","i297","i298","i295","i296","i299",])
	assert x1526.release_array("i256","i250","i255","i254","i257","i297","i298","i295","i296","i299",).shape == (3,3,1,1,5,1,5,4,1,2,)
	assert x1526.rank == 10
	x1503 = x321
	x1504 = x322
	x1505 = x1503 * x1504
	assert set(x1505.index_ids) == set(["i295","i306",])
	assert x1505.release_array("i295","i306",).shape == (4,2,)
	assert x1505.rank == 2
	x1506 = x323
	x1507 = x324
	x1508 = x1506 * x1507
	assert set(x1508.index_ids) == set(["i306","i296","i297","i298","i299",])
	assert x1508.release_array("i306","i296","i297","i298","i299",).shape == (2,1,1,5,2,)
	assert x1508.rank == 5
	x1509 = x1505 * x1508
	assert set(x1509.index_ids) == set(["i295","i296","i297","i298","i299",])
	assert x1509.release_array("i295","i296","i297","i298","i299",).shape == (4,1,1,5,2,)
	assert x1509.rank == 5
	x1525 = x1526 * x1509
	assert set(x1525.index_ids) == set(["i256","i250","i255","i254","i257",])
	assert x1525.release_array("i256","i250","i255","i254","i257",).shape == (3,3,1,1,5,)
	assert x1525.rank == 5
	x1527 = x1494 * x1525
	assert set(x1527.index_ids) == set(["i250","i254",])
	assert x1527.release_array("i250","i254",).shape == (3,1,)
	assert x1527.rank == 2
	x1403 = x333
	x1404 = x334
	x1405 = x1403 * x1404
	assert set(x1405.index_ids) == set(["i318","i317","i319","i316","i320",])
	assert x1405.release_array("i318","i317","i319","i316","i320",).shape == (1,3,5,1,5,)
	assert x1405.rank == 5
	x1406 = x335
	x1407 = x336
	x1408 = x1406 * x1407
	assert set(x1408.index_ids) == set(["i319","i320","i318",])
	assert x1408.release_array("i319","i320","i318",).shape == (5,5,1,)
	assert x1408.rank == 3
	x1409 = x1405 * x1408
	assert set(x1409.index_ids) == set(["i317","i316",])
	assert x1409.release_array("i317","i316",).shape == (3,1,)
	assert x1409.rank == 2
	x1410 = x339
	x1411 = x340
	x1412 = x341
	x1413 = x342
	x1414 = x1412 * x1413
	assert set(x1414.index_ids) == set(["i312","i324","i315","i317","i314",])
	assert x1414.release_array("i312","i324","i315","i317","i314",).shape == (2,2,4,3,4,)
	assert x1414.rank == 5
	x1416 = x1411 * x1414
	assert set(x1416.index_ids) == set(["i327","i316","i325","i326","i312","i315","i317","i314",])
	assert x1416.release_array("i327","i316","i325","i326","i312","i315","i317","i314",).shape == (3,1,1,3,2,4,3,4,)
	assert x1416.rank == 8
	x1415 = x1410 * x1416
	assert set(x1415.index_ids) == set(["i316","i312","i315","i317","i314",])
	assert x1415.release_array("i316","i312","i315","i317","i314",).shape == (1,2,4,3,4,)
	assert x1415.rank == 5
	x1417 = x1409 * x1415
	assert set(x1417.index_ids) == set(["i312","i315","i314",])
	assert x1417.release_array("i312","i315","i314",).shape == (2,4,4,)
	assert x1417.rank == 3
	x1418 = x347
	x1419 = x348
	x1420 = x1418 * x1419
	assert set(x1420.index_ids) == set(["i332","i331","i313","i314",])
	assert x1420.release_array("i332","i331","i313","i314",).shape == (4,1,4,4,)
	assert x1420.rank == 4
	x1421 = x349
	x1422 = x350
	x1423 = x1421 * x1422
	assert set(x1423.index_ids) == set(["i254","i331","i330","i329","i332",])
	assert x1423.release_array("i254","i331","i330","i329","i332",).shape == (1,1,2,1,4,)
	assert x1423.rank == 5
	x1424 = x353
	x1425 = x354
	x1426 = x1424 * x1425
	assert set(x1426.index_ids) == set(["i315","i335","i329","i336","i330",])
	assert x1426.release_array("i315","i335","i329","i336","i330",).shape == (4,5,1,1,2,)
	assert x1426.rank == 5
	x1427 = x355
	x1428 = x356
	x1429 = x1427 * x1428
	assert set(x1429.index_ids) == set(["i253","i336","i335",])
	assert x1429.release_array("i253","i336","i335",).shape == (3,1,5,)
	assert x1429.rank == 3
	x1430 = x1426 * x1429
	assert set(x1430.index_ids) == set(["i315","i329","i330","i253",])
	assert x1430.release_array("i315","i329","i330","i253",).shape == (4,1,2,3,)
	assert x1430.rank == 4
	x1432 = x1423 * x1430
	assert set(x1432.index_ids) == set(["i254","i331","i332","i315","i253",])
	assert x1432.release_array("i254","i331","i332","i315","i253",).shape == (1,1,4,4,3,)
	assert x1432.rank == 5
	x1431 = x1420 * x1432
	assert set(x1431.index_ids) == set(["i313","i314","i254","i315","i253",])
	assert x1431.release_array("i313","i314","i254","i315","i253",).shape == (4,4,1,4,3,)
	assert x1431.rank == 5
	x1433 = x1417 * x1431
	assert set(x1433.index_ids) == set(["i312","i313","i254","i253",])
	assert x1433.release_array("i312","i313","i254","i253",).shape == (2,4,1,3,)
	assert x1433.rank == 4
	x1529 = x1527 * x1433
	assert set(x1529.index_ids) == set(["i250","i312","i313","i253",])
	assert x1529.release_array("i250","i312","i313","i253",).shape == (3,2,4,3,)
	assert x1529.rank == 4
	x1434 = x363
	x1435 = x364
	x1436 = x1434 * x1435
	assert set(x1436.index_ids) == set(["i343","i342","i344","i252","i345",])
	assert x1436.release_array("i343","i342","i344","i252","i345",).shape == (5,1,1,1,2,)
	assert x1436.rank == 5
	x1437 = x365
	x1438 = x366
	x1439 = x1437 * x1438
	assert set(x1439.index_ids) == set(["i341","i345","i344",])
	assert x1439.release_array("i341","i345","i344",).shape == (4,2,1,)
	assert x1439.rank == 3
	x1440 = x1436 * x1439
	assert set(x1440.index_ids) == set(["i343","i342","i252","i341",])
	assert x1440.release_array("i343","i342","i252","i341",).shape == (5,1,1,4,)
	assert x1440.rank == 4
	x1441 = x369
	x1442 = x370
	x1443 = x1441 * x1442
	assert set(x1443.index_ids) == set(["i343","i313","i342","i251","i351",])
	assert x1443.release_array("i343","i313","i342","i251","i351",).shape == (5,4,1,5,3,)
	assert x1443.rank == 5
	x1444 = x371
	x1445 = x372
	x1446 = x1444 * x1445
	assert set(x1446.index_ids) == set(["i312","i351",])
	assert x1446.release_array("i312","i351",).shape == (2,3,)
	assert x1446.rank == 2
	x1447 = x1443 * x1446
	assert set(x1447.index_ids) == set(["i343","i313","i342","i251","i312",])
	assert x1447.release_array("i343","i313","i342","i251","i312",).shape == (5,4,1,5,2,)
	assert x1447.rank == 5
	x1448 = x377
	x1449 = x378
	x1450 = x379
	x1451 = x380
	x1452 = x1450 * x1451
	assert set(x1452.index_ids) == set(["i360","i359","i361","i358","i357",])
	assert x1452.release_array("i360","i359","i361","i358","i357",).shape == (2,3,1,1,1,)
	assert x1452.rank == 5
	x1454 = x1449 * x1452
	assert set(x1454.index_ids) == set(["i364","i362","i363","i360","i359","i361","i358","i357",])
	assert x1454.release_array("i364","i362","i363","i360","i359","i361","i358","i357",).shape == (1,4,1,2,3,1,1,1,)
	assert x1454.rank == 8
	x1453 = x1448 * x1454
	assert set(x1453.index_ids) == set(["i359","i358","i357",])
	assert x1453.release_array("i359","i358","i357",).shape == (3,1,1,)
	assert x1453.rank == 3
	x1457 = x383
	x1458 = x384
	x1459 = x1457 * x1458
	assert set(x1459.index_ids) == set(["i365","i341","i359","i357","i358",])
	assert x1459.release_array("i365","i341","i359","i357","i358",).shape == (2,4,3,1,1,)
	assert x1459.rank == 5
	x1455 = x385
	x1461 = x1459 * x1455
	assert set(x1461.index_ids) == set(["i365","i341","i359","i357","i358","i369","i368","i367",])
	assert x1461.release_array("i365","i341","i359","i357","i358","i369","i368","i367",).shape == (2,4,3,1,1,4,5,2,)
	assert x1461.rank == 8
	x1456 = x386
	x1460 = x1461 * x1456
	assert set(x1460.index_ids) == set(["i341","i359","i357","i358","i249",])
	assert x1460.release_array("i341","i359","i357","i358","i249",).shape == (4,3,1,1,5,)
	assert x1460.rank == 5
	x1462 = x1453 * x1460
	assert set(x1462.index_ids) == set(["i341","i249",])
	assert x1462.release_array("i341","i249",).shape == (4,5,)
	assert x1462.rank == 2
	x1464 = x1447 * x1462
	assert set(x1464.index_ids) == set(["i343","i313","i342","i251","i312","i341","i249",])
	assert x1464.release_array("i343","i313","i342","i251","i312","i341","i249",).shape == (5,4,1,5,2,4,5,)
	assert x1464.rank == 7
	x1463 = x1440 * x1464
	assert set(x1463.index_ids) == set(["i252","i313","i251","i312","i249",])
	assert x1463.release_array("i252","i313","i251","i312","i249",).shape == (1,4,5,2,5,)
	assert x1463.rank == 5
	x1528 = x1529 * x1463
	assert set(x1528.index_ids) == set(["i250","i253","i252","i251","i249",])
	assert x1528.release_array("i250","i253","i252","i251","i249",).shape == (3,3,1,5,5,)
	assert x1528.rank == 5
	x1307 = x397
	x1308 = x398
	x1309 = x1307 * x1308
	assert set(x1309.index_ids) == set(["i379","i377","i380","i378","i381",])
	assert x1309.release_array("i379","i377","i380","i378","i381",).shape == (3,4,2,2,2,)
	assert x1309.rank == 5
	x1310 = x399
	x1311 = x400
	x1312 = x1310 * x1311
	assert set(x1312.index_ids) == set(["i380","i381","i379","i378",])
	assert x1312.release_array("i380","i381","i379","i378",).shape == (2,2,3,2,)
	assert x1312.rank == 4
	x1313 = x1309 * x1312
	assert set(x1313.index_ids) == set(["i377",])
	assert x1313.release_array("i377",).shape == (4,)
	assert x1313.rank == 1
	x1314 = x403
	x1315 = x404
	x1316 = x405
	x1317 = x406
	x1318 = x1316 * x1317
	assert set(x1318.index_ids) == set(["i377","i376","i372","i375","i374",])
	assert x1318.release_array("i377","i376","i372","i375","i374",).shape == (4,2,3,2,3,)
	assert x1318.rank == 5
	x1320 = x1315 * x1318
	assert set(x1320.index_ids) == set(["i386","i387","i384","i385","i388","i377","i376","i372","i375","i374",])
	assert x1320.release_array("i386","i387","i384","i385","i388","i377","i376","i372","i375","i374",).shape == (4,5,2,5,1,4,2,3,2,3,)
	assert x1320.rank == 10
	x1319 = x1314 * x1320
	assert set(x1319.index_ids) == set(["i377","i376","i372","i375","i374",])
	assert x1319.release_array("i377","i376","i372","i375","i374",).shape == (4,2,3,2,3,)
	assert x1319.rank == 5
	x1323 = x411
	x1324 = x412
	x1325 = x1323 * x1324
	assert set(x1325.index_ids) == set([])
	assert x1325.release_array().shape == ()
	assert x1325.rank == 0
	x1321 = x413
	x1327 = x1325 * x1321
	assert set(x1327.index_ids) == set(["i397","i390","i398","i373",])
	assert x1327.release_array("i397","i390","i398","i373",).shape == (4,5,1,5,)
	assert x1327.rank == 4
	x1322 = x414
	x1326 = x1327 * x1322
	assert set(x1326.index_ids) == set(["i390","i373","i370","i375","i391",])
	assert x1326.release_array("i390","i373","i370","i375","i391",).shape == (5,5,5,2,1,)
	assert x1326.rank == 5
	x1330 = x417
	x1331 = x418
	x1332 = x1330 * x1331
	assert set(x1332.index_ids) == set(["i399","i391","i390","i400","i376",])
	assert x1332.release_array("i399","i391","i390","i400","i376",).shape == (3,1,5,3,2,)
	assert x1332.rank == 5
	x1328 = x419
	x1334 = x1332 * x1328
	assert set(x1334.index_ids) == set(["i391","i390","i400","i376","i403","i404","i405",])
	assert x1334.release_array("i391","i390","i400","i376","i403","i404","i405",).shape == (1,5,3,2,3,1,2,)
	assert x1334.rank == 7
	x1329 = x420
	x1333 = x1334 * x1329
	assert set(x1333.index_ids) == set(["i391","i390","i376","i371",])
	assert x1333.release_array("i391","i390","i376","i371",).shape == (1,5,2,2,)
	assert x1333.rank == 4
	x1335 = x1326 * x1333
	assert set(x1335.index_ids) == set(["i373","i370","i375","i376","i371",])
	assert x1335.release_array("i373","i370","i375","i376","i371",).shape == (5,5,2,2,2,)
	assert x1335.rank == 5
	x1337 = x1319 * x1335
	assert set(x1337.index_ids) == set(["i377","i372","i374","i373","i370","i371",])
	assert x1337.release_array("i377","i372","i374","i373","i370","i371",).shape == (4,3,3,5,5,2,)
	assert x1337.rank == 6
	x1336 = x1313 * x1337
	assert set(x1336.index_ids) == set(["i372","i374","i373","i370","i371",])
	assert x1336.release_array("i372","i374","i373","i370","i371",).shape == (3,3,5,5,2,)
	assert x1336.rank == 5
	x1285 = x427
	x1286 = x428
	x1287 = x1285 * x1286
	assert set(x1287.index_ids) == set([])
	assert x1287.release_array().shape == ()
	assert x1287.rank == 0
	x1283 = x429
	x1289 = x1287 * x1283
	assert set(x1289.index_ids) == set(["i411","i412","i371","i374","i372",])
	assert x1289.release_array("i411","i412","i371","i374","i372",).shape == (1,5,2,3,3,)
	assert x1289.rank == 5
	x1284 = x430
	x1288 = x1289 * x1284
	assert set(x1288.index_ids) == set(["i371","i374","i372","i370","i373",])
	assert x1288.release_array("i371","i374","i372","i370","i373",).shape == (2,3,3,5,5,)
	assert x1288.rank == 5
	x1277 = x433
	x1278 = x434
	x1279 = x1277 * x1278
	assert set(x1279.index_ids) == set(["i416","i414","i413","i417","i415",])
	assert x1279.release_array("i416","i414","i413","i417","i415",).shape == (3,3,4,4,2,)
	assert x1279.rank == 5
	x1291 = x1288 * x1279
	assert set(x1291.index_ids) == set(["i371","i374","i372","i370","i373","i416","i414","i413","i417","i415",])
	assert x1291.release_array("i371","i374","i372","i370","i373","i416","i414","i413","i417","i415",).shape == (2,3,3,5,5,3,3,4,4,2,)
	assert x1291.rank == 10
	x1280 = x435
	x1281 = x436
	x1282 = x1280 * x1281
	assert set(x1282.index_ids) == set(["i417","i414","i416","i413","i415",])
	assert x1282.release_array("i417","i414","i416","i413","i415",).shape == (4,3,3,4,2,)
	assert x1282.rank == 5
	x1290 = x1291 * x1282
	assert set(x1290.index_ids) == set(["i371","i374","i372","i370","i373",])
	assert x1290.release_array("i371","i374","i372","i370","i373",).shape == (2,3,3,5,5,)
	assert x1290.rank == 5
	x1339 = x1336 * x1290
	assert set(x1339.index_ids) == set([])
	assert x1339.release_array().shape == ()
	assert x1339.rank == 0
	x1292 = x441
	x1293 = x442
	x1294 = x1292 * x1293
	assert set(x1294.index_ids) == set(["i426","i422","i427","i428",])
	assert x1294.release_array("i426","i422","i427","i428",).shape == (4,2,3,3,)
	assert x1294.rank == 4
	x1295 = x443
	x1296 = x444
	x1297 = x1295 * x1296
	assert set(x1297.index_ids) == set(["i427","i428","i423","i425","i424",])
	assert x1297.release_array("i427","i428","i423","i425","i424",).shape == (3,3,1,3,5,)
	assert x1297.rank == 5
	x1298 = x447
	x1299 = x448
	x1300 = x449
	x1301 = x450
	x1302 = x1300 * x1301
	assert set(x1302.index_ids) == set([])
	assert x1302.release_array().shape == ()
	assert x1302.rank == 0
	x1304 = x1299 * x1302
	assert set(x1304.index_ids) == set(["i424","i433","i422","i432",])
	assert x1304.release_array("i424","i433","i422","i432",).shape == (5,5,2,4,)
	assert x1304.rank == 4
	x1303 = x1298 * x1304
	assert set(x1303.index_ids) == set(["i426","i425","i423","i424","i422",])
	assert x1303.release_array("i426","i425","i423","i424","i422",).shape == (4,3,1,5,2,)
	assert x1303.rank == 5
	x1306 = x1297 * x1303
	assert set(x1306.index_ids) == set(["i427","i428","i426","i422",])
	assert x1306.release_array("i427","i428","i426","i422",).shape == (3,3,4,2,)
	assert x1306.rank == 4
	x1305 = x1294 * x1306
	assert set(x1305.index_ids) == set([])
	assert x1305.release_array().shape == ()
	assert x1305.rank == 0
	x1338 = x1339 * x1305
	assert set(x1338.index_ids) == set([])
	assert x1338.release_array().shape == ()
	assert x1338.rank == 0
	x1531 = x1528 * x1338
	assert set(x1531.index_ids) == set(["i250","i253","i252","i251","i249",])
	assert x1531.release_array("i250","i253","i252","i251","i249",).shape == (3,3,1,5,5,)
	assert x1531.rank == 5
	x1354 = x459
	x1355 = x460
	x1356 = x1354 * x1355
	assert set(x1356.index_ids) == set(["i439","i251","i253","i249","i440",])
	assert x1356.release_array("i439","i251","i253","i249","i440",).shape == (2,5,3,5,3,)
	assert x1356.rank == 5
	x1357 = x461
	x1358 = x462
	x1359 = x1357 * x1358
	assert set(x1359.index_ids) == set([])
	assert x1359.release_array().shape == ()
	assert x1359.rank == 0
	x1360 = x465
	x1361 = x466
	x1362 = x467
	x1363 = x468
	x1364 = x1362 * x1363
	assert set(x1364.index_ids) == set(["i448","i449","i450",])
	assert x1364.release_array("i448","i449","i450",).shape == (1,1,4,)
	assert x1364.rank == 3
	x1366 = x1361 * x1364
	assert set(x1366.index_ids) == set(["i451","i448","i449",])
	assert x1366.release_array("i451","i448","i449",).shape == (5,1,1,)
	assert x1366.rank == 3
	x1365 = x1360 * x1366
	assert set(x1365.index_ids) == set(["i440","i250",])
	assert x1365.release_array("i440","i250",).shape == (3,3,)
	assert x1365.rank == 2
	x1368 = x1359 * x1365
	assert set(x1368.index_ids) == set(["i440","i250",])
	assert x1368.release_array("i440","i250",).shape == (3,3,)
	assert x1368.rank == 2
	x1367 = x1356 * x1368
	assert set(x1367.index_ids) == set(["i439","i251","i253","i249","i250",])
	assert x1367.release_array("i439","i251","i253","i249","i250",).shape == (2,5,3,5,3,)
	assert x1367.rank == 5
	x1340 = x473
	x1341 = x474
	x1342 = x475
	x1343 = x476
	x1344 = x1342 * x1343
	assert set(x1344.index_ids) == set(["i459","i458",])
	assert x1344.release_array("i459","i458",).shape == (3,2,)
	assert x1344.rank == 2
	x1346 = x1341 * x1344
	assert set(x1346.index_ids) == set(["i457","i460","i459","i458",])
	assert x1346.release_array("i457","i460","i459","i458",).shape == (1,5,3,2,)
	assert x1346.rank == 4
	x1345 = x1340 * x1346
	assert set(x1345.index_ids) == set(["i456","i439","i455","i457","i458",])
	assert x1345.release_array("i456","i439","i455","i457","i458",).shape == (2,2,1,1,2,)
	assert x1345.rank == 5
	x1370 = x1367 * x1345
	assert set(x1370.index_ids) == set(["i251","i253","i249","i250","i456","i455","i457","i458",])
	assert x1370.release_array("i251","i253","i249","i250","i456","i455","i457","i458",).shape == (5,3,5,3,2,1,1,2,)
	assert x1370.rank == 8
	x1349 = x479
	x1350 = x480
	x1351 = x1349 * x1350
	assert set(x1351.index_ids) == set([])
	assert x1351.release_array().shape == ()
	assert x1351.rank == 0
	x1347 = x481
	x1353 = x1351 * x1347
	assert set(x1353.index_ids) == set([])
	assert x1353.release_array().shape == ()
	assert x1353.rank == 0
	x1348 = x482
	x1352 = x1353 * x1348
	assert set(x1352.index_ids) == set(["i252","i455","i457","i456","i458",])
	assert x1352.release_array("i252","i455","i457","i456","i458",).shape == (1,1,1,2,2,)
	assert x1352.rank == 5
	x1369 = x1370 * x1352
	assert set(x1369.index_ids) == set(["i251","i253","i249","i250","i252",])
	assert x1369.release_array("i251","i253","i249","i250","i252",).shape == (5,3,5,3,1,)
	assert x1369.rank == 5
	x1371 = x489
	x1372 = x490
	x1373 = x491
	x1374 = x492
	x1375 = x1373 * x1374
	assert set(x1375.index_ids) == set(["i476","i473","i475","i477","i474",])
	assert x1375.release_array("i476","i473","i475","i477","i474",).shape == (2,5,2,4,1,)
	assert x1375.rank == 5
	x1377 = x1372 * x1375
	assert set(x1377.index_ids) == set(["i479","i478","i480","i473","i477","i474",])
	assert x1377.release_array("i479","i478","i480","i473","i477","i474",).shape == (1,5,1,5,4,1,)
	assert x1377.rank == 6
	x1376 = x1371 * x1377
	assert set(x1376.index_ids) == set(["i473","i474",])
	assert x1376.release_array("i473","i474",).shape == (5,1,)
	assert x1376.rank == 2
	x1378 = x495
	x1379 = x496
	x1380 = x497
	x1381 = x498
	x1382 = x1380 * x1381
	assert set(x1382.index_ids) == set(["i471","i472","i469","i474","i470",])
	assert x1382.release_array("i471","i472","i469","i474","i470",).shape == (5,4,5,1,4,)
	assert x1382.rank == 5
	x1384 = x1379 * x1382
	assert set(x1384.index_ids) == set(["i481","i484","i485","i483","i482","i471","i472","i469","i474","i470",])
	assert x1384.release_array("i481","i484","i485","i483","i482","i471","i472","i469","i474","i470",).shape == (5,5,5,1,4,5,4,5,1,4,)
	assert x1384.rank == 10
	x1383 = x1378 * x1384
	assert set(x1383.index_ids) == set(["i471","i472","i469","i474","i470",])
	assert x1383.release_array("i471","i472","i469","i474","i470",).shape == (5,4,5,1,4,)
	assert x1383.rank == 5
	x1385 = x1376 * x1383
	assert set(x1385.index_ids) == set(["i473","i471","i472","i469","i470",])
	assert x1385.release_array("i473","i471","i472","i469","i470",).shape == (5,5,4,5,4,)
	assert x1385.rank == 5
	x1386 = x503
	x1387 = x504
	x1388 = x1386 * x1387
	assert set(x1388.index_ids) == set([])
	assert x1388.release_array().shape == ()
	assert x1388.rank == 0
	x1389 = x505
	x1390 = x506
	x1391 = x1389 * x1390
	assert set(x1391.index_ids) == set(["i473","i472","i471","i469","i470",])
	assert x1391.release_array("i473","i472","i471","i469","i470",).shape == (5,4,5,5,4,)
	assert x1391.rank == 5
	x1392 = x509
	x1393 = x510
	x1394 = x1392 * x1393
	assert set(x1394.index_ids) == set(["i498","i495","i494","i496","i497",])
	assert x1394.release_array("i498","i495","i494","i496","i497",).shape == (4,4,5,2,4,)
	assert x1394.rank == 5
	x1395 = x511
	x1396 = x512
	x1397 = x1395 * x1396
	assert set(x1397.index_ids) == set(["i495","i496","i494","i497","i498",])
	assert x1397.release_array("i495","i496","i494","i497","i498",).shape == (4,2,5,4,4,)
	assert x1397.rank == 5
	x1398 = x1394 * x1397
	assert set(x1398.index_ids) == set([])
	assert x1398.release_array().shape == ()
	assert x1398.rank == 0
	x1400 = x1391 * x1398
	assert set(x1400.index_ids) == set(["i473","i472","i471","i469","i470",])
	assert x1400.release_array("i473","i472","i471","i469","i470",).shape == (5,4,5,5,4,)
	assert x1400.rank == 5
	x1399 = x1388 * x1400
	assert set(x1399.index_ids) == set(["i473","i472","i471","i469","i470",])
	assert x1399.release_array("i473","i472","i471","i469","i470",).shape == (5,4,5,5,4,)
	assert x1399.rank == 5
	x1401 = x1385 * x1399
	assert set(x1401.index_ids) == set([])
	assert x1401.release_array().shape == ()
	assert x1401.rank == 0
	x1402 = x1369 * x1401
	assert set(x1402.index_ids) == set(["i251","i253","i249","i250","i252",])
	assert x1402.release_array("i251","i253","i249","i250","i252",).shape == (5,3,5,3,1,)
	assert x1402.rank == 5
	x1530 = x1531 * x1402
	assert set(x1530.index_ids) == set([])
	assert x1530.release_array().shape == ()
	assert x1530.rank == 0
	x1533 = x1275 * x1530
	assert set(x1533.index_ids) == set([])
	assert x1533.release_array().shape == ()
	assert x1533.rank == 0
	x1532 = x1148 * x1533
	assert set(x1532.index_ids) == set(["i5","i2","i1","i6","i0",])
	assert x1532.release_array("i5","i2","i1","i6","i0",).shape == (3,3,2,4,3,)
	assert x1532.rank == 5
	x1566 = x527
	x1567 = x528
	x1568 = x1566 * x1567
	assert set(x1568.index_ids) == set(["i512","i513","i510","i508","i511",])
	assert x1568.release_array("i512","i513","i510","i508","i511",).shape == (5,2,1,3,1,)
	assert x1568.rank == 5
	x1564 = x529
	x1570 = x1568 * x1564
	assert set(x1570.index_ids) == set(["i512","i513","i510","i508","i511","i516","i515","i519","i517","i518",])
	assert x1570.release_array("i512","i513","i510","i508","i511","i516","i515","i519","i517","i518",).shape == (5,2,1,3,1,2,2,5,5,3,)
	assert x1570.rank == 10
	x1565 = x530
	x1569 = x1570 * x1565
	assert set(x1569.index_ids) == set(["i512","i513","i510","i508","i511",])
	assert x1569.release_array("i512","i513","i510","i508","i511",).shape == (5,2,1,3,1,)
	assert x1569.rank == 5
	x1571 = x533
	x1572 = x534
	x1573 = x535
	x1574 = x536
	x1575 = x1573 * x1574
	assert set(x1575.index_ids) == set(["i512","i520","i510","i521","i509",])
	assert x1575.release_array("i512","i520","i510","i521","i509",).shape == (5,1,1,3,5,)
	assert x1575.rank == 5
	x1577 = x1572 * x1575
	assert set(x1577.index_ids) == set(["i522","i523","i513","i512","i520","i510","i521","i509",])
	assert x1577.release_array("i522","i523","i513","i512","i520","i510","i521","i509",).shape == (2,4,2,5,1,1,3,5,)
	assert x1577.rank == 8
	x1576 = x1571 * x1577
	assert set(x1576.index_ids) == set(["i511","i513","i512","i510","i509",])
	assert x1576.release_array("i511","i513","i512","i510","i509",).shape == (1,2,5,1,5,)
	assert x1576.rank == 5
	x1578 = x1569 * x1576
	assert set(x1578.index_ids) == set(["i508","i509",])
	assert x1578.release_array("i508","i509",).shape == (3,5,)
	assert x1578.rank == 2
	x1581 = x541
	x1582 = x542
	x1583 = x1581 * x1582
	assert set(x1583.index_ids) == set(["i527","i528","i505","i526",])
	assert x1583.release_array("i527","i528","i505","i526",).shape == (2,4,3,3,)
	assert x1583.rank == 4
	x1579 = x543
	x1585 = x1583 * x1579
	assert set(x1585.index_ids) == set(["i505","i526","i506","i507","i525",])
	assert x1585.release_array("i505","i526","i506","i507","i525",).shape == (3,3,4,1,5,)
	assert x1585.rank == 5
	x1580 = x544
	x1584 = x1585 * x1580
	assert set(x1584.index_ids) == set(["i505","i526","i506","i507","i525",])
	assert x1584.release_array("i505","i526","i506","i507","i525",).shape == (3,3,4,1,5,)
	assert x1584.rank == 5
	x1588 = x547
	x1589 = x548
	x1590 = x1588 * x1589
	assert set(x1590.index_ids) == set(["i532",])
	assert x1590.release_array("i532",).shape == (4,)
	assert x1590.rank == 1
	x1586 = x549
	x1592 = x1590 * x1586
	assert set(x1592.index_ids) == set(["i526","i537","i525","i509",])
	assert x1592.release_array("i526","i537","i525","i509",).shape == (3,4,5,5,)
	assert x1592.rank == 4
	x1587 = x550
	x1591 = x1592 * x1587
	assert set(x1591.index_ids) == set(["i526","i525","i509","i504",])
	assert x1591.release_array("i526","i525","i509","i504",).shape == (3,5,5,4,)
	assert x1591.rank == 4
	x1593 = x1584 * x1591
	assert set(x1593.index_ids) == set(["i505","i506","i507","i509","i504",])
	assert x1593.release_array("i505","i506","i507","i509","i504",).shape == (3,4,1,5,4,)
	assert x1593.rank == 5
	x1594 = x1578 * x1593
	assert set(x1594.index_ids) == set(["i508","i505","i506","i507","i504",])
	assert x1594.release_array("i508","i505","i506","i507","i504",).shape == (3,3,4,1,4,)
	assert x1594.rank == 5
	x1536 = x557
	x1537 = x558
	x1538 = x1536 * x1537
	assert set(x1538.index_ids) == set(["i545","i546","i544",])
	assert x1538.release_array("i545","i546","i544",).shape == (2,4,2,)
	assert x1538.rank == 3
	x1534 = x559
	x1540 = x1538 * x1534
	assert set(x1540.index_ids) == set(["i546","i544","i551","i543","i550",])
	assert x1540.release_array("i546","i544","i551","i543","i550",).shape == (4,2,4,2,5,)
	assert x1540.rank == 5
	x1535 = x560
	x1539 = x1540 * x1535
	assert set(x1539.index_ids) == set(["i543","i539",])
	assert x1539.release_array("i543","i539",).shape == (2,1,)
	assert x1539.rank == 2
	x1541 = x563
	x1542 = x564
	x1543 = x1541 * x1542
	assert set(x1543.index_ids) == set(["i553","i552","i543","i538",])
	assert x1543.release_array("i553","i552","i543","i538",).shape == (1,3,2,3,)
	assert x1543.rank == 4
	x1544 = x565
	x1545 = x566
	x1546 = x1544 * x1545
	assert set(x1546.index_ids) == set(["i552","i541","i540","i542","i553",])
	assert x1546.release_array("i552","i541","i540","i542","i553",).shape == (3,4,2,5,1,)
	assert x1546.rank == 5
	x1547 = x1543 * x1546
	assert set(x1547.index_ids) == set(["i543","i538","i541","i540","i542",])
	assert x1547.release_array("i543","i538","i541","i540","i542",).shape == (2,3,4,2,5,)
	assert x1547.rank == 5
	x1548 = x1539 * x1547
	assert set(x1548.index_ids) == set(["i539","i538","i541","i540","i542",])
	assert x1548.release_array("i539","i538","i541","i540","i542",).shape == (1,3,4,2,5,)
	assert x1548.rank == 5
	x1596 = x1594 * x1548
	assert set(x1596.index_ids) == set(["i508","i505","i506","i507","i504","i539","i538","i541","i540","i542",])
	assert x1596.release_array("i508","i505","i506","i507","i504","i539","i538","i541","i540","i542",).shape == (3,3,4,1,4,1,3,4,2,5,)
	assert x1596.rank == 10
	x1549 = x571
	x1550 = x572
	x1551 = x1549 * x1550
	assert set(x1551.index_ids) == set(["i557","i542","i560","i559","i558",])
	assert x1551.release_array("i557","i542","i560","i559","i558",).shape == (1,5,2,4,4,)
	assert x1551.rank == 5
	x1552 = x573
	x1553 = x574
	x1554 = x1552 * x1553
	assert set(x1554.index_ids) == set(["i559","i558","i560",])
	assert x1554.release_array("i559","i558","i560",).shape == (4,4,2,)
	assert x1554.rank == 3
	x1555 = x577
	x1556 = x578
	x1557 = x579
	x1558 = x580
	x1559 = x1557 * x1558
	assert set(x1559.index_ids) == set(["i557","i566",])
	assert x1559.release_array("i557","i566",).shape == (1,5,)
	assert x1559.rank == 2
	x1561 = x1556 * x1559
	assert set(x1561.index_ids) == set(["i539","i567","i540","i541","i557",])
	assert x1561.release_array("i539","i567","i540","i541","i557",).shape == (1,5,2,4,1,)
	assert x1561.rank == 5
	x1560 = x1555 * x1561
	assert set(x1560.index_ids) == set(["i538","i539","i540","i541","i557",])
	assert x1560.release_array("i538","i539","i540","i541","i557",).shape == (3,1,2,4,1,)
	assert x1560.rank == 5
	x1563 = x1554 * x1560
	assert set(x1563.index_ids) == set(["i559","i558","i560","i538","i539","i540","i541","i557",])
	assert x1563.release_array("i559","i558","i560","i538","i539","i540","i541","i557",).shape == (4,4,2,3,1,2,4,1,)
	assert x1563.rank == 8
	x1562 = x1551 * x1563
	assert set(x1562.index_ids) == set(["i542","i538","i539","i540","i541",])
	assert x1562.release_array("i542","i538","i539","i540","i541",).shape == (5,3,1,2,4,)
	assert x1562.rank == 5
	x1595 = x1596 * x1562
	assert set(x1595.index_ids) == set(["i508","i505","i506","i507","i504",])
	assert x1595.release_array("i508","i505","i506","i507","i504",).shape == (3,3,4,1,4,)
	assert x1595.rank == 5
	x1597 = x589
	x1598 = x590
	x1599 = x591
	x1600 = x592
	x1601 = x1599 * x1600
	assert set(x1601.index_ids) == set(["i573","i577","i575","i574","i576",])
	assert x1601.release_array("i573","i577","i575","i574","i576",).shape == (3,1,5,5,3,)
	assert x1601.rank == 5
	x1603 = x1598 * x1601
	assert set(x1603.index_ids) == set(["i581","i580","i578","i579","i582","i573","i577","i575","i574","i576",])
	assert x1603.release_array("i581","i580","i578","i579","i582","i573","i577","i575","i574","i576",).shape == (1,1,3,4,5,3,1,5,5,3,)
	assert x1603.rank == 10
	x1602 = x1597 * x1603
	assert set(x1602.index_ids) == set(["i573","i577","i575","i574","i576",])
	assert x1602.release_array("i573","i577","i575","i574","i576",).shape == (3,1,5,5,3,)
	assert x1602.rank == 5
	x1604 = x595
	x1605 = x596
	x1606 = x1604 * x1605
	assert set(x1606.index_ids) == set(["i585","i586","i576","i577","i587",])
	assert x1606.release_array("i585","i586","i576","i577","i587",).shape == (5,5,3,1,2,)
	assert x1606.rank == 5
	x1607 = x597
	x1608 = x598
	x1609 = x1607 * x1608
	assert set(x1609.index_ids) == set(["i585","i586","i587",])
	assert x1609.release_array("i585","i586","i587",).shape == (5,5,2,)
	assert x1609.rank == 3
	x1610 = x1606 * x1609
	assert set(x1610.index_ids) == set(["i576","i577",])
	assert x1610.release_array("i576","i577",).shape == (3,1,)
	assert x1610.rank == 2
	x1613 = x603
	x1614 = x604
	x1615 = x1613 * x1614
	assert set(x1615.index_ids) == set(["i593","i592",])
	assert x1615.release_array("i593","i592",).shape == (1,5,)
	assert x1615.rank == 2
	x1611 = x605
	x1617 = x1615 * x1611
	assert set(x1617.index_ids) == set(["i593","i592",])
	assert x1617.release_array("i593","i592",).shape == (1,5,)
	assert x1617.rank == 2
	x1612 = x606
	x1616 = x1617 * x1612
	assert set(x1616.index_ids) == set(["i592","i571","i574","i575","i573",])
	assert x1616.release_array("i592","i571","i574","i575","i573",).shape == (5,3,5,5,3,)
	assert x1616.rank == 5
	x1618 = x609
	x1619 = x610
	x1620 = x1618 * x1619
	assert set(x1620.index_ids) == set(["i592","i572","i599","i597","i598",])
	assert x1620.release_array("i592","i572","i599","i597","i598",).shape == (5,4,3,5,2,)
	assert x1620.rank == 5
	x1621 = x611
	x1622 = x612
	x1623 = x1621 * x1622
	assert set(x1623.index_ids) == set(["i599","i597","i598",])
	assert x1623.release_array("i599","i597","i598",).shape == (3,5,2,)
	assert x1623.rank == 3
	x1624 = x1620 * x1623
	assert set(x1624.index_ids) == set(["i592","i572",])
	assert x1624.release_array("i592","i572",).shape == (5,4,)
	assert x1624.rank == 2
	x1625 = x1616 * x1624
	assert set(x1625.index_ids) == set(["i571","i574","i575","i573","i572",])
	assert x1625.release_array("i571","i574","i575","i573","i572",).shape == (3,5,5,3,4,)
	assert x1625.rank == 5
	x1627 = x1610 * x1625
	assert set(x1627.index_ids) == set(["i576","i577","i571","i574","i575","i573","i572",])
	assert x1627.release_array("i576","i577","i571","i574","i575","i573","i572",).shape == (3,1,3,5,5,3,4,)
	assert x1627.rank == 7
	x1626 = x1602 * x1627
	assert set(x1626.index_ids) == set(["i571","i572",])
	assert x1626.release_array("i571","i572",).shape == (3,4,)
	assert x1626.rank == 2
	x1628 = x619
	x1629 = x620
	x1630 = x1628 * x1629
	assert set(x1630.index_ids) == set(["i609","i604","i610","i606",])
	assert x1630.release_array("i609","i604","i610","i606",).shape == (4,3,5,3,)
	assert x1630.rank == 4
	x1631 = x621
	x1632 = x622
	x1633 = x1631 * x1632
	assert set(x1633.index_ids) == set(["i607","i503","i610","i608","i609",])
	assert x1633.release_array("i607","i503","i610","i608","i609",).shape == (2,5,5,2,4,)
	assert x1633.rank == 5
	x1634 = x625
	x1635 = x626
	x1636 = x627
	x1637 = x628
	x1638 = x1636 * x1637
	assert set(x1638.index_ids) == set(["i605","i608","i607","i606","i571",])
	assert x1638.release_array("i605","i608","i607","i606","i571",).shape == (4,2,2,3,3,)
	assert x1638.rank == 5
	x1640 = x1635 * x1638
	assert set(x1640.index_ids) == set(["i614","i613","i616","i617","i615","i605","i608","i607","i606","i571",])
	assert x1640.release_array("i614","i613","i616","i617","i615","i605","i608","i607","i606","i571",).shape == (1,2,4,2,2,4,2,2,3,3,)
	assert x1640.rank == 10
	x1639 = x1634 * x1640
	assert set(x1639.index_ids) == set(["i605","i608","i607","i606","i571",])
	assert x1639.release_array("i605","i608","i607","i606","i571",).shape == (4,2,2,3,3,)
	assert x1639.rank == 5
	x1642 = x1633 * x1639
	assert set(x1642.index_ids) == set(["i503","i610","i609","i605","i606","i571",])
	assert x1642.release_array("i503","i610","i609","i605","i606","i571",).shape == (5,5,4,4,3,3,)
	assert x1642.rank == 6
	x1641 = x1630 * x1642
	assert set(x1641.index_ids) == set(["i604","i503","i605","i571",])
	assert x1641.release_array("i604","i503","i605","i571",).shape == (3,5,4,3,)
	assert x1641.rank == 4
	x1643 = x633
	x1644 = x634
	x1645 = x1643 * x1644
	assert set(x1645.index_ids) == set(["i622","i621","i620","i623","i619",])
	assert x1645.release_array("i622","i621","i620","i623","i619",).shape == (4,3,3,4,4,)
	assert x1645.rank == 5
	x1646 = x635
	x1647 = x636
	x1648 = x1646 * x1647
	assert set(x1648.index_ids) == set(["i622","i623","i619","i621","i620",])
	assert x1648.release_array("i622","i623","i619","i621","i620",).shape == (4,4,4,3,3,)
	assert x1648.rank == 5
	x1649 = x639
	x1650 = x640
	x1651 = x641
	x1652 = x642
	x1653 = x1651 * x1652
	assert set(x1653.index_ids) == set([])
	assert x1653.release_array().shape == ()
	assert x1653.rank == 0
	x1655 = x1650 * x1653
	assert set(x1655.index_ids) == set(["i572","i604","i508","i507","i605",])
	assert x1655.release_array("i572","i604","i508","i507","i605",).shape == (4,3,3,1,4,)
	assert x1655.rank == 5
	x1654 = x1649 * x1655
	assert set(x1654.index_ids) == set(["i572","i604","i508","i507","i605",])
	assert x1654.release_array("i572","i604","i508","i507","i605",).shape == (4,3,3,1,4,)
	assert x1654.rank == 5
	x1657 = x1648 * x1654
	assert set(x1657.index_ids) == set(["i622","i623","i619","i621","i620","i572","i604","i508","i507","i605",])
	assert x1657.release_array("i622","i623","i619","i621","i620","i572","i604","i508","i507","i605",).shape == (4,4,4,3,3,4,3,3,1,4,)
	assert x1657.rank == 10
	x1656 = x1645 * x1657
	assert set(x1656.index_ids) == set(["i572","i604","i508","i507","i605",])
	assert x1656.release_array("i572","i604","i508","i507","i605",).shape == (4,3,3,1,4,)
	assert x1656.rank == 5
	x1658 = x1641 * x1656
	assert set(x1658.index_ids) == set(["i503","i571","i572","i508","i507",])
	assert x1658.release_array("i503","i571","i572","i508","i507",).shape == (5,3,4,3,1,)
	assert x1658.rank == 5
	x1659 = x1626 * x1658
	assert set(x1659.index_ids) == set(["i503","i508","i507",])
	assert x1659.release_array("i503","i508","i507",).shape == (5,3,1,)
	assert x1659.rank == 3
	x1660 = x1595 * x1659
	assert set(x1660.index_ids) == set(["i505","i506","i504","i503",])
	assert x1660.release_array("i505","i506","i504","i503",).shape == (3,4,4,5,)
	assert x1660.rank == 4
	x1737 = x653
	x1738 = x654
	x1739 = x1737 * x1738
	assert set(x1739.index_ids) == set(["i641","i643","i642",])
	assert x1739.release_array("i641","i643","i642",).shape == (1,4,4,)
	assert x1739.rank == 3
	x1740 = x655
	x1741 = x656
	x1742 = x1740 * x1741
	assert set(x1742.index_ids) == set(["i637","i643","i641","i642","i640",])
	assert x1742.release_array("i637","i643","i641","i642","i640",).shape == (2,4,1,4,2,)
	assert x1742.rank == 5
	x1743 = x659
	x1744 = x660
	x1745 = x661
	x1746 = x662
	x1747 = x1745 * x1746
	assert set(x1747.index_ids) == set(["i647","i638",])
	assert x1747.release_array("i647","i638",).shape == (3,2,)
	assert x1747.rank == 2
	x1749 = x1744 * x1747
	assert set(x1749.index_ids) == set(["i635","i640","i648","i649","i639","i647","i638",])
	assert x1749.release_array("i635","i640","i648","i649","i639","i647","i638",).shape == (5,2,4,1,4,3,2,)
	assert x1749.rank == 7
	x1748 = x1743 * x1749
	assert set(x1748.index_ids) == set(["i636","i635","i640","i639","i638",])
	assert x1748.release_array("i636","i635","i640","i639","i638",).shape == (3,5,2,4,2,)
	assert x1748.rank == 5
	x1751 = x1742 * x1748
	assert set(x1751.index_ids) == set(["i637","i643","i641","i642","i636","i635","i639","i638",])
	assert x1751.release_array("i637","i643","i641","i642","i636","i635","i639","i638",).shape == (2,4,1,4,3,5,4,2,)
	assert x1751.rank == 8
	x1750 = x1739 * x1751
	assert set(x1750.index_ids) == set(["i637","i636","i635","i639","i638",])
	assert x1750.release_array("i637","i636","i635","i639","i638",).shape == (2,3,5,4,2,)
	assert x1750.rank == 5
	x1723 = x667
	x1724 = x668
	x1725 = x669
	x1726 = x670
	x1727 = x1725 * x1726
	assert set(x1727.index_ids) == set(["i656","i638","i654","i653","i655",])
	assert x1727.release_array("i656","i638","i654","i653","i655",).shape == (1,2,2,5,3,)
	assert x1727.rank == 5
	x1729 = x1724 * x1727
	assert set(x1729.index_ids) == set(["i658","i657","i659","i638","i654","i653",])
	assert x1729.release_array("i658","i657","i659","i638","i654","i653",).shape == (1,2,1,2,2,5,)
	assert x1729.rank == 6
	x1728 = x1723 * x1729
	assert set(x1728.index_ids) == set(["i638","i654","i653",])
	assert x1728.release_array("i638","i654","i653",).shape == (2,2,5,)
	assert x1728.rank == 3
	x1753 = x1750 * x1728
	assert set(x1753.index_ids) == set(["i637","i636","i635","i639","i654","i653",])
	assert x1753.release_array("i637","i636","i635","i639","i654","i653",).shape == (2,3,5,4,2,5,)
	assert x1753.rank == 6
	x1730 = x673
	x1731 = x674
	x1732 = x675
	x1733 = x676
	x1734 = x1732 * x1733
	assert set(x1734.index_ids) == set(["i663","i633","i634","i662","i654",])
	assert x1734.release_array("i663","i633","i634","i662","i654",).shape == (4,5,2,1,2,)
	assert x1734.rank == 5
	x1736 = x1731 * x1734
	assert set(x1736.index_ids) == set(["i665","i664","i633","i634","i662","i654",])
	assert x1736.release_array("i665","i664","i633","i634","i662","i654",).shape == (1,5,5,2,1,2,)
	assert x1736.rank == 6
	x1735 = x1730 * x1736
	assert set(x1735.index_ids) == set(["i653","i639","i633","i634","i654",])
	assert x1735.release_array("i653","i639","i633","i634","i654",).shape == (5,4,5,2,2,)
	assert x1735.rank == 5
	x1752 = x1753 * x1735
	assert set(x1752.index_ids) == set(["i637","i636","i635","i633","i634",])
	assert x1752.release_array("i637","i636","i635","i633","i634",).shape == (2,3,5,5,2,)
	assert x1752.rank == 5
	x1754 = x683
	x1755 = x684
	x1756 = x685
	x1757 = x686
	x1758 = x1756 * x1757
	assert set(x1758.index_ids) == set(["i668","i671","i673","i669","i672",])
	assert x1758.release_array("i668","i671","i673","i669","i672",).shape == (5,1,3,2,3,)
	assert x1758.rank == 5
	x1760 = x1755 * x1758
	assert set(x1760.index_ids) == set(["i674","i668","i671","i673","i669","i672",])
	assert x1760.release_array("i674","i668","i671","i673","i669","i672",).shape == (2,5,1,3,2,3,)
	assert x1760.rank == 6
	x1759 = x1754 * x1760
	assert set(x1759.index_ids) == set(["i670","i668","i669",])
	assert x1759.release_array("i670","i668","i669",).shape == (1,5,2,)
	assert x1759.rank == 3
	x1763 = x689
	x1764 = x690
	x1765 = x1763 * x1764
	assert set(x1765.index_ids) == set(["i669","i668","i667","i636","i670",])
	assert x1765.release_array("i669","i668","i667","i636","i670",).shape == (2,5,5,3,1,)
	assert x1765.rank == 5
	x1761 = x691
	x1767 = x1765 * x1761
	assert set(x1767.index_ids) == set(["i669","i668","i667","i636","i670","i680","i676","i679","i677","i678",])
	assert x1767.release_array("i669","i668","i667","i636","i670","i680","i676","i679","i677","i678",).shape == (2,5,5,3,1,1,4,4,5,3,)
	assert x1767.rank == 10
	x1762 = x692
	x1766 = x1767 * x1762
	assert set(x1766.index_ids) == set(["i669","i668","i667","i636","i670",])
	assert x1766.release_array("i669","i668","i667","i636","i670",).shape == (2,5,5,3,1,)
	assert x1766.rank == 5
	x1768 = x697
	x1769 = x698
	x1770 = x1768 * x1769
	assert set(x1770.index_ids) == set(["i637","i505","i667","i634","i635",])
	assert x1770.release_array("i637","i505","i667","i634","i635",).shape == (2,3,5,2,5,)
	assert x1770.rank == 5
	x1771 = x699
	x1772 = x700
	x1773 = x1771 * x1772
	assert set(x1773.index_ids) == set([])
	assert x1773.release_array().shape == ()
	assert x1773.rank == 0
	x1774 = x703
	x1775 = x704
	x1776 = x705
	x1777 = x706
	x1778 = x1776 * x1777
	assert set(x1778.index_ids) == set(["i692","i690","i688","i691","i689",])
	assert x1778.release_array("i692","i690","i688","i691","i689",).shape == (5,5,1,4,3,)
	assert x1778.rank == 5
	x1780 = x1775 * x1778
	assert set(x1780.index_ids) == set(["i694","i693","i692","i688","i691",])
	assert x1780.release_array("i694","i693","i692","i688","i691",).shape == (4,3,5,1,4,)
	assert x1780.rank == 5
	x1779 = x1774 * x1780
	assert set(x1779.index_ids) == set([])
	assert x1779.release_array().shape == ()
	assert x1779.rank == 0
	x1782 = x1773 * x1779
	assert set(x1782.index_ids) == set([])
	assert x1782.release_array().shape == ()
	assert x1782.rank == 0
	x1781 = x1770 * x1782
	assert set(x1781.index_ids) == set(["i637","i505","i667","i634","i635",])
	assert x1781.release_array("i637","i505","i667","i634","i635",).shape == (2,3,5,2,5,)
	assert x1781.rank == 5
	x1784 = x1766 * x1781
	assert set(x1784.index_ids) == set(["i669","i668","i636","i670","i637","i505","i634","i635",])
	assert x1784.release_array("i669","i668","i636","i670","i637","i505","i634","i635",).shape == (2,5,3,1,2,3,2,5,)
	assert x1784.rank == 8
	x1783 = x1759 * x1784
	assert set(x1783.index_ids) == set(["i636","i637","i505","i634","i635",])
	assert x1783.release_array("i636","i637","i505","i634","i635",).shape == (3,2,3,2,5,)
	assert x1783.rank == 5
	x1785 = x1752 * x1783
	assert set(x1785.index_ids) == set(["i633","i505",])
	assert x1785.release_array("i633","i505",).shape == (5,3,)
	assert x1785.rank == 2
	x1681 = x715
	x1682 = x716
	x1683 = x1681 * x1682
	assert set(x1683.index_ids) == set(["i702","i700",])
	assert x1683.release_array("i702","i700",).shape == (5,1,)
	assert x1683.rank == 2
	x1684 = x717
	x1685 = x718
	x1686 = x1684 * x1685
	assert set(x1686.index_ids) == set(["i698","i699","i702","i696","i701",])
	assert x1686.release_array("i698","i699","i702","i696","i701",).shape == (3,4,5,3,2,)
	assert x1686.rank == 5
	x1687 = x1683 * x1686
	assert set(x1687.index_ids) == set(["i700","i698","i699","i696","i701",])
	assert x1687.release_array("i700","i698","i699","i696","i701",).shape == (1,3,4,3,2,)
	assert x1687.rank == 5
	x1675 = x721
	x1676 = x722
	x1677 = x1675 * x1676
	assert set(x1677.index_ids) == set(["i701","i709","i708",])
	assert x1677.release_array("i701","i709","i708",).shape == (2,1,4,)
	assert x1677.rank == 3
	x1689 = x1687 * x1677
	assert set(x1689.index_ids) == set(["i700","i698","i699","i696","i709","i708",])
	assert x1689.release_array("i700","i698","i699","i696","i709","i708",).shape == (1,3,4,3,1,4,)
	assert x1689.rank == 6
	x1678 = x723
	x1679 = x724
	x1680 = x1678 * x1679
	assert set(x1680.index_ids) == set(["i697","i709","i708","i699","i700",])
	assert x1680.release_array("i697","i709","i708","i699","i700",).shape == (3,1,4,4,1,)
	assert x1680.rank == 5
	x1688 = x1689 * x1680
	assert set(x1688.index_ids) == set(["i698","i696","i697",])
	assert x1688.release_array("i698","i696","i697",).shape == (3,3,3,)
	assert x1688.rank == 3
	x1661 = x729
	x1662 = x730
	x1663 = x1661 * x1662
	assert set(x1663.index_ids) == set(["i715","i716","i718","i717","i714",])
	assert x1663.release_array("i715","i716","i718","i717","i714",).shape == (5,1,1,3,1,)
	assert x1663.rank == 5
	x1664 = x731
	x1665 = x732
	x1666 = x1664 * x1665
	assert set(x1666.index_ids) == set(["i718","i715","i714","i716","i717",])
	assert x1666.release_array("i718","i715","i714","i716","i717",).shape == (1,5,1,1,3,)
	assert x1666.rank == 5
	x1667 = x1663 * x1666
	assert set(x1667.index_ids) == set([])
	assert x1667.release_array().shape == ()
	assert x1667.rank == 0
	x1691 = x1688 * x1667
	assert set(x1691.index_ids) == set(["i698","i696","i697",])
	assert x1691.release_array("i698","i696","i697",).shape == (3,3,3,)
	assert x1691.rank == 3
	x1668 = x735
	x1669 = x736
	x1670 = x1668 * x1669
	assert set(x1670.index_ids) == set(["i506","i719","i720","i695","i697",])
	assert x1670.release_array("i506","i719","i720","i695","i697",).shape == (4,2,1,1,3,)
	assert x1670.rank == 5
	x1671 = x737
	x1672 = x738
	x1673 = x1671 * x1672
	assert set(x1673.index_ids) == set(["i719","i696","i720","i698",])
	assert x1673.release_array("i719","i696","i720","i698",).shape == (2,3,1,3,)
	assert x1673.rank == 4
	x1674 = x1670 * x1673
	assert set(x1674.index_ids) == set(["i506","i695","i697","i696","i698",])
	assert x1674.release_array("i506","i695","i697","i696","i698",).shape == (4,1,3,3,3,)
	assert x1674.rank == 5
	x1690 = x1691 * x1674
	assert set(x1690.index_ids) == set(["i506","i695",])
	assert x1690.release_array("i506","i695",).shape == (4,1,)
	assert x1690.rank == 2
	x1787 = x1785 * x1690
	assert set(x1787.index_ids) == set(["i633","i505","i506","i695",])
	assert x1787.release_array("i633","i505","i506","i695",).shape == (5,3,4,1,)
	assert x1787.rank == 4
	x1692 = x745
	x1693 = x746
	x1694 = x747
	x1695 = x748
	x1696 = x1694 * x1695
	assert set(x1696.index_ids) == set(["i728","i727","i726","i729",])
	assert x1696.release_array("i728","i727","i726","i729",).shape == (3,4,3,2,)
	assert x1696.rank == 4
	x1698 = x1693 * x1696
	assert set(x1698.index_ids) == set(["i731","i730","i727","i726",])
	assert x1698.release_array("i731","i730","i727","i726",).shape == (2,5,4,3,)
	assert x1698.rank == 4
	x1697 = x1692 * x1698
	assert set(x1697.index_ids) == set(["i725",])
	assert x1697.release_array("i725",).shape == (1,)
	assert x1697.rank == 1
	x1699 = x751
	x1700 = x752
	x1701 = x1699 * x1700
	assert set(x1701.index_ids) == set([])
	assert x1701.release_array().shape == ()
	assert x1701.rank == 0
	x1702 = x753
	x1703 = x754
	x1704 = x1702 * x1703
	assert set(x1704.index_ids) == set(["i695","i725","i4","i724","i723",])
	assert x1704.release_array("i695","i725","i4","i724","i723",).shape == (1,1,3,5,5,)
	assert x1704.rank == 5
	x1705 = x1701 * x1704
	assert set(x1705.index_ids) == set(["i695","i725","i4","i724","i723",])
	assert x1705.release_array("i695","i725","i4","i724","i723",).shape == (1,1,3,5,5,)
	assert x1705.rank == 5
	x1706 = x1697 * x1705
	assert set(x1706.index_ids) == set(["i695","i4","i724","i723",])
	assert x1706.release_array("i695","i4","i724","i723",).shape == (1,3,5,5,)
	assert x1706.rank == 4
	x1713 = x759
	x1714 = x760
	x1715 = x761
	x1716 = x762
	x1717 = x1715 * x1716
	assert set(x1717.index_ids) == set(["i633","i742",])
	assert x1717.release_array("i633","i742",).shape == (5,4,)
	assert x1717.rank == 2
	x1719 = x1714 * x1717
	assert set(x1719.index_ids) == set(["i6","i744","i743","i723","i633","i742",])
	assert x1719.release_array("i6","i744","i743","i723","i633","i742",).shape == (4,5,4,5,5,4,)
	assert x1719.rank == 6
	x1718 = x1713 * x1719
	assert set(x1718.index_ids) == set(["i741","i740","i6","i723","i633",])
	assert x1718.release_array("i741","i740","i6","i723","i633",).shape == (1,1,4,5,5,)
	assert x1718.rank == 5
	x1707 = x765
	x1708 = x766
	x1709 = x1707 * x1708
	assert set(x1709.index_ids) == set(["i740","i748","i724","i749","i750",])
	assert x1709.release_array("i740","i748","i724","i749","i750",).shape == (1,4,5,2,3,)
	assert x1709.rank == 5
	x1721 = x1718 * x1709
	assert set(x1721.index_ids) == set(["i741","i6","i723","i633","i748","i724","i749","i750",])
	assert x1721.release_array("i741","i6","i723","i633","i748","i724","i749","i750",).shape == (1,4,5,5,4,5,2,3,)
	assert x1721.rank == 8
	x1710 = x767
	x1711 = x768
	x1712 = x1710 * x1711
	assert set(x1712.index_ids) == set(["i748","i5","i750","i741","i749",])
	assert x1712.release_array("i748","i5","i750","i741","i749",).shape == (4,3,3,1,2,)
	assert x1712.rank == 5
	x1720 = x1721 * x1712
	assert set(x1720.index_ids) == set(["i6","i723","i633","i724","i5",])
	assert x1720.release_array("i6","i723","i633","i724","i5",).shape == (4,5,5,5,3,)
	assert x1720.rank == 5
	x1722 = x1706 * x1720
	assert set(x1722.index_ids) == set(["i695","i4","i6","i633","i5",])
	assert x1722.release_array("i695","i4","i6","i633","i5",).shape == (1,3,4,5,3,)
	assert x1722.rank == 5
	x1786 = x1787 * x1722
	assert set(x1786.index_ids) == set(["i505","i506","i4","i6","i5",])
	assert x1786.release_array("i505","i506","i4","i6","i5",).shape == (3,4,3,4,3,)
	assert x1786.rank == 5
	x1880 = x781
	x1881 = x782
	x1882 = x1880 * x1881
	assert set(x1882.index_ids) == set(["i763","i766","i765","i764","i762",])
	assert x1882.release_array("i763","i766","i765","i764","i762",).shape == (2,2,1,5,2,)
	assert x1882.rank == 5
	x1883 = x783
	x1884 = x784
	x1885 = x1883 * x1884
	assert set(x1885.index_ids) == set(["i766","i765","i763","i761","i764",])
	assert x1885.release_array("i766","i765","i763","i761","i764",).shape == (2,1,2,1,5,)
	assert x1885.rank == 5
	x1886 = x787
	x1887 = x788
	x1888 = x789
	x1889 = x790
	x1890 = x1888 * x1889
	assert set(x1890.index_ids) == set(["i759","i756","i3","i760","i762",])
	assert x1890.release_array("i759","i756","i3","i760","i762",).shape == (2,1,5,2,2,)
	assert x1890.rank == 5
	x1892 = x1887 * x1890
	assert set(x1892.index_ids) == set(["i771","i768","i769","i770","i772","i759","i756","i3","i760","i762",])
	assert x1892.release_array("i771","i768","i769","i770","i772","i759","i756","i3","i760","i762",).shape == (2,2,2,5,2,2,1,5,2,2,)
	assert x1892.rank == 10
	x1891 = x1886 * x1892
	assert set(x1891.index_ids) == set(["i759","i756","i3","i760","i762",])
	assert x1891.release_array("i759","i756","i3","i760","i762",).shape == (2,1,5,2,2,)
	assert x1891.rank == 5
	x1894 = x1885 * x1891
	assert set(x1894.index_ids) == set(["i766","i765","i763","i761","i764","i759","i756","i3","i760","i762",])
	assert x1894.release_array("i766","i765","i763","i761","i764","i759","i756","i3","i760","i762",).shape == (2,1,2,1,5,2,1,5,2,2,)
	assert x1894.rank == 10
	x1893 = x1882 * x1894
	assert set(x1893.index_ids) == set(["i761","i759","i756","i3","i760",])
	assert x1893.release_array("i761","i759","i756","i3","i760",).shape == (1,2,1,5,2,)
	assert x1893.rank == 5
	x1897 = x795
	x1898 = x796
	x1899 = x1897 * x1898
	assert set(x1899.index_ids) == set(["i779","i778","i776","i777","i775",])
	assert x1899.release_array("i779","i778","i776","i777","i775",).shape == (3,5,2,1,3,)
	assert x1899.rank == 5
	x1895 = x797
	x1901 = x1899 * x1895
	assert set(x1901.index_ids) == set(["i779","i778","i776","i777","i775","i781","i780","i782",])
	assert x1901.release_array("i779","i778","i776","i777","i775","i781","i780","i782",).shape == (3,5,2,1,3,2,3,1,)
	assert x1901.rank == 8
	x1896 = x798
	x1900 = x1901 * x1896
	assert set(x1900.index_ids) == set(["i776","i777","i775",])
	assert x1900.release_array("i776","i777","i775",).shape == (2,1,3,)
	assert x1900.rank == 3
	x1904 = x801
	x1905 = x802
	x1906 = x1904 * x1905
	assert set(x1906.index_ids) == set(["i783","i775","i757","i776","i761",])
	assert x1906.release_array("i783","i775","i757","i776","i761",).shape == (4,3,2,2,1,)
	assert x1906.rank == 5
	x1902 = x803
	x1908 = x1906 * x1902
	assert set(x1908.index_ids) == set(["i775","i757","i776","i761","i786","i777","i785","i787",])
	assert x1908.release_array("i775","i757","i776","i761","i786","i777","i785","i787",).shape == (3,2,2,1,3,1,1,1,)
	assert x1908.rank == 8
	x1903 = x804
	x1907 = x1908 * x1903
	assert set(x1907.index_ids) == set(["i775","i757","i776","i761","i777",])
	assert x1907.release_array("i775","i757","i776","i761","i777",).shape == (3,2,2,1,1,)
	assert x1907.rank == 5
	x1909 = x1900 * x1907
	assert set(x1909.index_ids) == set(["i757","i761",])
	assert x1909.release_array("i757","i761",).shape == (2,1,)
	assert x1909.rank == 2
	x1910 = x1893 * x1909
	assert set(x1910.index_ids) == set(["i759","i756","i3","i760","i757",])
	assert x1910.release_array("i759","i756","i3","i760","i757",).shape == (2,1,5,2,2,)
	assert x1910.rank == 5
	x1850 = x811
	x1851 = x812
	x1852 = x1850 * x1851
	assert set(x1852.index_ids) == set([])
	assert x1852.release_array().shape == ()
	assert x1852.rank == 0
	x1853 = x813
	x1854 = x814
	x1855 = x1853 * x1854
	assert set(x1855.index_ids) == set(["i760","i790","i789","i788","i759",])
	assert x1855.release_array("i760","i790","i789","i788","i759",).shape == (2,2,4,3,2,)
	assert x1855.rank == 5
	x1856 = x1852 * x1855
	assert set(x1856.index_ids) == set(["i760","i790","i789","i788","i759",])
	assert x1856.release_array("i760","i790","i789","i788","i759",).shape == (2,2,4,3,2,)
	assert x1856.rank == 5
	x1859 = x817
	x1860 = x818
	x1861 = x1859 * x1860
	assert set(x1861.index_ids) == set(["i799","i798","i790","i796","i797",])
	assert x1861.release_array("i799","i798","i790","i796","i797",).shape == (4,1,2,4,3,)
	assert x1861.rank == 5
	x1857 = x819
	x1863 = x1861 * x1857
	assert set(x1863.index_ids) == set(["i790","i755",])
	assert x1863.release_array("i790","i755",).shape == (2,4,)
	assert x1863.rank == 2
	x1858 = x820
	x1862 = x1863 * x1858
	assert set(x1862.index_ids) == set(["i790","i755",])
	assert x1862.release_array("i790","i755",).shape == (2,4,)
	assert x1862.rank == 2
	x1864 = x1856 * x1862
	assert set(x1864.index_ids) == set(["i760","i789","i788","i759","i755",])
	assert x1864.release_array("i760","i789","i788","i759","i755",).shape == (2,4,3,2,4,)
	assert x1864.rank == 5
	x1912 = x1910 * x1864
	assert set(x1912.index_ids) == set(["i756","i3","i757","i789","i788","i755",])
	assert x1912.release_array("i756","i3","i757","i789","i788","i755",).shape == (1,5,2,4,3,4,)
	assert x1912.rank == 6
	x1871 = x825
	x1872 = x826
	x1873 = x1871 * x1872
	assert set(x1873.index_ids) == set(["i801","i804","i800","i803","i789",])
	assert x1873.release_array("i801","i804","i800","i803","i789",).shape == (5,2,4,1,4,)
	assert x1873.rank == 5
	x1874 = x827
	x1875 = x828
	x1876 = x1874 * x1875
	assert set(x1876.index_ids) == set(["i803","i804","i788","i802",])
	assert x1876.release_array("i803","i804","i788","i802",).shape == (1,2,3,4,)
	assert x1876.rank == 4
	x1877 = x1873 * x1876
	assert set(x1877.index_ids) == set(["i801","i800","i789","i788","i802",])
	assert x1877.release_array("i801","i800","i789","i788","i802",).shape == (5,4,4,3,4,)
	assert x1877.rank == 5
	x1865 = x831
	x1866 = x832
	x1867 = x1865 * x1866
	assert set(x1867.index_ids) == set(["i808",])
	assert x1867.release_array("i808",).shape == (3,)
	assert x1867.rank == 1
	x1879 = x1877 * x1867
	assert set(x1879.index_ids) == set(["i801","i800","i789","i788","i802","i808",])
	assert x1879.release_array("i801","i800","i789","i788","i802","i808",).shape == (5,4,4,3,4,3,)
	assert x1879.rank == 6
	x1868 = x833
	x1869 = x834
	x1870 = x1868 * x1869
	assert set(x1870.index_ids) == set(["i808","i800","i758","i801","i802",])
	assert x1870.release_array("i808","i800","i758","i801","i802",).shape == (3,4,5,5,4,)
	assert x1870.rank == 5
	x1878 = x1879 * x1870
	assert set(x1878.index_ids) == set(["i789","i788","i758",])
	assert x1878.release_array("i789","i788","i758",).shape == (4,3,5,)
	assert x1878.rank == 3
	x1911 = x1912 * x1878
	assert set(x1911.index_ids) == set(["i756","i3","i757","i755","i758",])
	assert x1911.release_array("i756","i3","i757","i755","i758",).shape == (1,5,2,4,5,)
	assert x1911.rank == 5
	x1794 = x843
	x1795 = x844
	x1796 = x845
	x1797 = x846
	x1798 = x1796 * x1797
	assert set(x1798.index_ids) == set(["i816","i817","i815","i504","i818",])
	assert x1798.release_array("i816","i817","i815","i504","i818",).shape == (2,4,3,4,5,)
	assert x1798.rank == 5
	x1800 = x1795 * x1798
	assert set(x1800.index_ids) == set(["i820","i819","i814","i817","i815","i504",])
	assert x1800.release_array("i820","i819","i814","i817","i815","i504",).shape == (1,4,3,4,3,4,)
	assert x1800.rank == 6
	x1799 = x1794 * x1800
	assert set(x1799.index_ids) == set(["i814","i504",])
	assert x1799.release_array("i814","i504",).shape == (3,4,)
	assert x1799.rank == 2
	x1788 = x849
	x1789 = x850
	x1790 = x1788 * x1789
	assert set(x1790.index_ids) == set(["i814","i824","i813","i823","i758",])
	assert x1790.release_array("i814","i824","i813","i823","i758",).shape == (3,4,1,3,5,)
	assert x1790.rank == 5
	x1802 = x1799 * x1790
	assert set(x1802.index_ids) == set(["i504","i824","i813","i823","i758",])
	assert x1802.release_array("i504","i824","i813","i823","i758",).shape == (4,4,1,3,5,)
	assert x1802.rank == 5
	x1791 = x851
	x1792 = x852
	x1793 = x1791 * x1792
	assert set(x1793.index_ids) == set(["i823","i754","i824","i757",])
	assert x1793.release_array("i823","i754","i824","i757",).shape == (3,2,4,2,)
	assert x1793.rank == 4
	x1801 = x1802 * x1793
	assert set(x1801.index_ids) == set(["i504","i813","i758","i754","i757",])
	assert x1801.release_array("i504","i813","i758","i754","i757",).shape == (4,1,5,2,2,)
	assert x1801.rank == 5
	x1803 = x857
	x1804 = x858
	x1805 = x1803 * x1804
	assert set(x1805.index_ids) == set([])
	assert x1805.release_array().shape == ()
	assert x1805.rank == 0
	x1806 = x859
	x1807 = x860
	x1808 = x1806 * x1807
	assert set(x1808.index_ids) == set(["i829","i830","i831","i832","i828",])
	assert x1808.release_array("i829","i830","i831","i832","i828",).shape == (4,4,4,5,1,)
	assert x1808.rank == 5
	x1809 = x863
	x1810 = x864
	x1811 = x1809 * x1810
	assert set(x1811.index_ids) == set([])
	assert x1811.release_array().shape == ()
	assert x1811.rank == 0
	x1812 = x865
	x1813 = x866
	x1814 = x1812 * x1813
	assert set(x1814.index_ids) == set(["i831","i828","i832","i829","i830",])
	assert x1814.release_array("i831","i828","i832","i829","i830",).shape == (4,1,5,4,4,)
	assert x1814.rank == 5
	x1815 = x1811 * x1814
	assert set(x1815.index_ids) == set(["i831","i828","i832","i829","i830",])
	assert x1815.release_array("i831","i828","i832","i829","i830",).shape == (4,1,5,4,4,)
	assert x1815.rank == 5
	x1817 = x1808 * x1815
	assert set(x1817.index_ids) == set([])
	assert x1817.release_array().shape == ()
	assert x1817.rank == 0
	x1816 = x1805 * x1817
	assert set(x1816.index_ids) == set([])
	assert x1816.release_array().shape == ()
	assert x1816.rank == 0
	x1818 = x1801 * x1816
	assert set(x1818.index_ids) == set(["i504","i813","i758","i754","i757",])
	assert x1818.release_array("i504","i813","i758","i754","i757",).shape == (4,1,5,2,2,)
	assert x1818.rank == 5
	x1914 = x1911 * x1818
	assert set(x1914.index_ids) == set(["i756","i3","i755","i504","i813","i754",])
	assert x1914.release_array("i756","i3","i755","i504","i813","i754",).shape == (1,5,4,4,1,2,)
	assert x1914.rank == 6
	x1821 = x873
	x1822 = x874
	x1823 = x1821 * x1822
	assert set(x1823.index_ids) == set(["i852","i853","i851",])
	assert x1823.release_array("i852","i853","i851",).shape == (4,1,5,)
	assert x1823.rank == 3
	x1819 = x875
	x1825 = x1823 * x1819
	assert set(x1825.index_ids) == set(["i852","i853","i851","i849","i856",])
	assert x1825.release_array("i852","i853","i851","i849","i856",).shape == (4,1,5,3,5,)
	assert x1825.rank == 5
	x1820 = x876
	x1824 = x1825 * x1820
	assert set(x1824.index_ids) == set(["i849","i850",])
	assert x1824.release_array("i849","i850",).shape == (3,4,)
	assert x1824.rank == 2
	x1828 = x879
	x1829 = x880
	x1830 = x1828 * x1829
	assert set(x1830.index_ids) == set(["i850","i857",])
	assert x1830.release_array("i850","i857",).shape == (4,1,)
	assert x1830.rank == 2
	x1826 = x881
	x1832 = x1830 * x1826
	assert set(x1832.index_ids) == set(["i850","i846","i848","i813","i847",])
	assert x1832.release_array("i850","i846","i848","i813","i847",).shape == (4,5,2,1,5,)
	assert x1832.rank == 5
	x1827 = x882
	x1831 = x1832 * x1827
	assert set(x1831.index_ids) == set(["i850","i846","i848","i813","i847",])
	assert x1831.release_array("i850","i846","i848","i813","i847",).shape == (4,5,2,1,5,)
	assert x1831.rank == 5
	x1833 = x1824 * x1831
	assert set(x1833.index_ids) == set(["i849","i846","i848","i813","i847",])
	assert x1833.release_array("i849","i846","i848","i813","i847",).shape == (3,5,2,1,5,)
	assert x1833.rank == 5
	x1840 = x887
	x1841 = x888
	x1842 = x1840 * x1841
	assert set(x1842.index_ids) == set(["i848","i847","i849","i846","i862",])
	assert x1842.release_array("i848","i847","i849","i846","i862",).shape == (2,5,3,5,5,)
	assert x1842.rank == 5
	x1843 = x889
	x1844 = x890
	x1845 = x1843 * x1844
	assert set(x1845.index_ids) == set(["i861","i862",])
	assert x1845.release_array("i861","i862",).shape == (1,5,)
	assert x1845.rank == 2
	x1846 = x1842 * x1845
	assert set(x1846.index_ids) == set(["i848","i847","i849","i846","i861",])
	assert x1846.release_array("i848","i847","i849","i846","i861",).shape == (2,5,3,5,1,)
	assert x1846.rank == 5
	x1834 = x893
	x1835 = x894
	x1836 = x1834 * x1835
	assert set(x1836.index_ids) == set(["i867","i869","i866","i868","i861",])
	assert x1836.release_array("i867","i869","i866","i868","i861",).shape == (3,5,2,2,1,)
	assert x1836.rank == 5
	x1848 = x1846 * x1836
	assert set(x1848.index_ids) == set(["i848","i847","i849","i846","i867","i869","i866","i868",])
	assert x1848.release_array("i848","i847","i849","i846","i867","i869","i866","i868",).shape == (2,5,3,5,3,5,2,2,)
	assert x1848.rank == 8
	x1837 = x895
	x1838 = x896
	x1839 = x1837 * x1838
	assert set(x1839.index_ids) == set(["i866","i868","i867","i869",])
	assert x1839.release_array("i866","i868","i867","i869",).shape == (2,2,3,5,)
	assert x1839.rank == 4
	x1847 = x1848 * x1839
	assert set(x1847.index_ids) == set(["i848","i847","i849","i846",])
	assert x1847.release_array("i848","i847","i849","i846",).shape == (2,5,3,5,)
	assert x1847.rank == 4
	x1849 = x1833 * x1847
	assert set(x1849.index_ids) == set(["i813",])
	assert x1849.release_array("i813",).shape == (1,)
	assert x1849.rank == 1
	x1913 = x1914 * x1849
	assert set(x1913.index_ids) == set(["i756","i3","i755","i504","i754",])
	assert x1913.release_array("i756","i3","i755","i504","i754",).shape == (1,5,4,4,2,)
	assert x1913.rank == 5
	x1937 = x907
	x1938 = x908
	x1939 = x1937 * x1938
	assert set(x1939.index_ids) == set(["i879","i876","i875","i880",])
	assert x1939.release_array("i879","i876","i875","i880",).shape == (2,5,5,5,)
	assert x1939.rank == 4
	x1935 = x909
	x1941 = x1939 * x1935
	assert set(x1941.index_ids) == set(["i876","i875","i878","i877","i874",])
	assert x1941.release_array("i876","i875","i878","i877","i874",).shape == (5,5,2,4,4,)
	assert x1941.rank == 5
	x1936 = x910
	x1940 = x1941 * x1936
	assert set(x1940.index_ids) == set(["i876","i875","i878","i877","i874",])
	assert x1940.release_array("i876","i875","i878","i877","i874",).shape == (5,5,2,4,4,)
	assert x1940.rank == 5
	x1929 = x913
	x1930 = x914
	x1931 = x1929 * x1930
	assert set(x1931.index_ids) == set(["i885","i887","i878","i884","i886",])
	assert x1931.release_array("i885","i887","i878","i884","i886",).shape == (3,2,2,1,1,)
	assert x1931.rank == 5
	x1943 = x1940 * x1931
	assert set(x1943.index_ids) == set(["i876","i875","i877","i874","i885","i887","i884","i886",])
	assert x1943.release_array("i876","i875","i877","i874","i885","i887","i884","i886",).shape == (5,5,4,4,3,2,1,1,)
	assert x1943.rank == 8
	x1932 = x915
	x1933 = x916
	x1934 = x1932 * x1933
	assert set(x1934.index_ids) == set(["i877","i884","i886","i887","i885",])
	assert x1934.release_array("i877","i884","i886","i887","i885",).shape == (4,1,1,2,3,)
	assert x1934.rank == 5
	x1942 = x1943 * x1934
	assert set(x1942.index_ids) == set(["i876","i875","i874",])
	assert x1942.release_array("i876","i875","i874",).shape == (5,5,4,)
	assert x1942.rank == 3
	x1915 = x921
	x1916 = x922
	x1917 = x923
	x1918 = x924
	x1919 = x1917 * x1918
	assert set(x1919.index_ids) == set(["i873","i891",])
	assert x1919.release_array("i873","i891",).shape == (3,2,)
	assert x1919.rank == 2
	x1921 = x1916 * x1919
	assert set(x1921.index_ids) == set(["i872","i875","i876","i874","i873",])
	assert x1921.release_array("i872","i875","i876","i874","i873",).shape == (2,5,5,4,3,)
	assert x1921.rank == 5
	x1920 = x1915 * x1921
	assert set(x1920.index_ids) == set(["i872","i875","i876","i874","i873",])
	assert x1920.release_array("i872","i875","i876","i874","i873",).shape == (2,5,5,4,3,)
	assert x1920.rank == 5
	x1945 = x1942 * x1920
	assert set(x1945.index_ids) == set(["i872","i873",])
	assert x1945.release_array("i872","i873",).shape == (2,3,)
	assert x1945.rank == 2
	x1922 = x927
	x1923 = x928
	x1924 = x1922 * x1923
	assert set(x1924.index_ids) == set(["i897","i895","i898","i896","i899",])
	assert x1924.release_array("i897","i895","i898","i896","i899",).shape == (4,2,2,2,2,)
	assert x1924.rank == 5
	x1925 = x929
	x1926 = x930
	x1927 = x1925 * x1926
	assert set(x1927.index_ids) == set(["i895","i899","i896","i897","i898",])
	assert x1927.release_array("i895","i899","i896","i897","i898",).shape == (2,2,2,4,2,)
	assert x1927.rank == 5
	x1928 = x1924 * x1927
	assert set(x1928.index_ids) == set([])
	assert x1928.release_array().shape == ()
	assert x1928.rank == 0
	x1944 = x1945 * x1928
	assert set(x1944.index_ids) == set(["i872","i873",])
	assert x1944.release_array("i872","i873",).shape == (2,3,)
	assert x1944.rank == 2
	x1946 = x937
	x1947 = x938
	x1948 = x939
	x1949 = x940
	x1950 = x1948 * x1949
	assert set(x1950.index_ids) == set(["i903","i906","i904","i907","i905",])
	assert x1950.release_array("i903","i906","i904","i907","i905",).shape == (5,5,5,4,3,)
	assert x1950.rank == 5
	x1952 = x1947 * x1950
	assert set(x1952.index_ids) == set(["i903","i906","i904","i907","i905",])
	assert x1952.release_array("i903","i906","i904","i907","i905",).shape == (5,5,5,4,3,)
	assert x1952.rank == 5
	x1951 = x1946 * x1952
	assert set(x1951.index_ids) == set([])
	assert x1951.release_array().shape == ()
	assert x1951.rank == 0
	x1953 = x943
	x1954 = x944
	x1955 = x945
	x1956 = x946
	x1957 = x1955 * x1956
	assert set(x1957.index_ids) == set(["i908","i756",])
	assert x1957.release_array("i908","i756",).shape == (4,1,)
	assert x1957.rank == 2
	x1959 = x1954 * x1957
	assert set(x1959.index_ids) == set(["i754","i755","i873","i503","i756",])
	assert x1959.release_array("i754","i755","i873","i503","i756",).shape == (2,4,3,5,1,)
	assert x1959.rank == 5
	x1958 = x1953 * x1959
	assert set(x1958.index_ids) == set(["i754","i755","i873","i503","i756",])
	assert x1958.release_array("i754","i755","i873","i503","i756",).shape == (2,4,3,5,1,)
	assert x1958.rank == 5
	x1968 = x951
	x1969 = x952
	x1970 = x1968 * x1969
	assert set(x1970.index_ids) == set(["i916","i913","i917","i915","i914",])
	assert x1970.release_array("i916","i913","i917","i915","i914",).shape == (1,2,5,4,2,)
	assert x1970.rank == 5
	x1966 = x953
	x1972 = x1970 * x1966
	assert set(x1972.index_ids) == set(["i916","i913","i917","i915","i914","i924","i923","i920","i921","i922",])
	assert x1972.release_array("i916","i913","i917","i915","i914","i924","i923","i920","i921","i922",).shape == (1,2,5,4,2,3,2,4,4,1,)
	assert x1972.rank == 10
	x1967 = x954
	x1971 = x1972 * x1967
	assert set(x1971.index_ids) == set(["i916","i913","i917","i915","i914",])
	assert x1971.release_array("i916","i913","i917","i915","i914",).shape == (1,2,5,4,2,)
	assert x1971.rank == 5
	x1960 = x957
	x1961 = x958
	x1962 = x1960 * x1961
	assert set(x1962.index_ids) == set(["i925","i914",])
	assert x1962.release_array("i925","i914",).shape == (3,2,)
	assert x1962.rank == 2
	x1974 = x1971 * x1962
	assert set(x1974.index_ids) == set(["i916","i913","i917","i915","i925",])
	assert x1974.release_array("i916","i913","i917","i915","i925",).shape == (1,2,5,4,3,)
	assert x1974.rank == 5
	x1963 = x959
	x1964 = x960
	x1965 = x1963 * x1964
	assert set(x1965.index_ids) == set(["i913","i915","i916","i917","i925",])
	assert x1965.release_array("i913","i915","i916","i917","i925",).shape == (2,4,1,5,3,)
	assert x1965.rank == 5
	x1973 = x1974 * x1965
	assert set(x1973.index_ids) == set([])
	assert x1973.release_array().shape == ()
	assert x1973.rank == 0
	x1976 = x1958 * x1973
	assert set(x1976.index_ids) == set(["i754","i755","i873","i503","i756",])
	assert x1976.release_array("i754","i755","i873","i503","i756",).shape == (2,4,3,5,1,)
	assert x1976.rank == 5
	x1975 = x1951 * x1976
	assert set(x1975.index_ids) == set(["i754","i755","i873","i503","i756",])
	assert x1975.release_array("i754","i755","i873","i503","i756",).shape == (2,4,3,5,1,)
	assert x1975.rank == 5
	x1977 = x969
	x1978 = x970
	x1979 = x971
	x1980 = x972
	x1981 = x1979 * x1980
	assert set(x1981.index_ids) == set(["i935","i937","i936","i938","i939",])
	assert x1981.release_array("i935","i937","i936","i938","i939",).shape == (4,1,5,3,2,)
	assert x1981.rank == 5
	x1983 = x1978 * x1981
	assert set(x1983.index_ids) == set(["i940","i935","i937","i936","i939",])
	assert x1983.release_array("i940","i935","i937","i936","i939",).shape == (2,4,1,5,2,)
	assert x1983.rank == 5
	x1982 = x1977 * x1983
	assert set(x1982.index_ids) == set([])
	assert x1982.release_array().shape == ()
	assert x1982.rank == 0
	x1986 = x975
	x1987 = x976
	x1988 = x1986 * x1987
	assert set(x1988.index_ids) == set([])
	assert x1988.release_array().shape == ()
	assert x1988.rank == 0
	x1984 = x977
	x1990 = x1988 * x1984
	assert set(x1990.index_ids) == set(["i946","i947","i932","i934",])
	assert x1990.release_array("i946","i947","i932","i934",).shape == (3,5,4,1,)
	assert x1990.rank == 4
	x1985 = x978
	x1989 = x1990 * x1985
	assert set(x1989.index_ids) == set(["i932","i934","i931","i933","i930",])
	assert x1989.release_array("i932","i934","i931","i933","i930",).shape == (4,1,1,1,2,)
	assert x1989.rank == 5
	x1991 = x1982 * x1989
	assert set(x1991.index_ids) == set(["i932","i934","i931","i933","i930",])
	assert x1991.release_array("i932","i934","i931","i933","i930",).shape == (4,1,1,1,2,)
	assert x1991.rank == 5
	x1992 = x983
	x1993 = x984
	x1994 = x1992 * x1993
	assert set(x1994.index_ids) == set(["i949","i952","i948","i951",])
	assert x1994.release_array("i949","i952","i948","i951",).shape == (4,1,1,5,)
	assert x1994.rank == 4
	x1995 = x985
	x1996 = x986
	x1997 = x1995 * x1996
	assert set(x1997.index_ids) == set(["i951","i952","i872","i934","i950",])
	assert x1997.release_array("i951","i952","i872","i934","i950",).shape == (5,1,2,1,5,)
	assert x1997.rank == 5
	x1998 = x1994 * x1997
	assert set(x1998.index_ids) == set(["i949","i948","i872","i934","i950",])
	assert x1998.release_array("i949","i948","i872","i934","i950",).shape == (4,1,2,1,5,)
	assert x1998.rank == 5
	x1999 = x989
	x2000 = x990
	x2001 = x991
	x2002 = x992
	x2003 = x2001 * x2002
	assert set(x2003.index_ids) == set(["i956","i957",])
	assert x2003.release_array("i956","i957",).shape == (3,5,)
	assert x2003.rank == 2
	x2005 = x2000 * x2003
	assert set(x2005.index_ids) == set(["i956","i957",])
	assert x2005.release_array("i956","i957",).shape == (3,5,)
	assert x2005.rank == 2
	x2004 = x1999 * x2005
	assert set(x2004.index_ids) == set(["i950","i949","i948",])
	assert x2004.release_array("i950","i949","i948",).shape == (5,4,1,)
	assert x2004.rank == 3
	x2006 = x1998 * x2004
	assert set(x2006.index_ids) == set(["i872","i934",])
	assert x2006.release_array("i872","i934",).shape == (2,1,)
	assert x2006.rank == 2
	x2007 = x1991 * x2006
	assert set(x2007.index_ids) == set(["i932","i931","i933","i930","i872",])
	assert x2007.release_array("i932","i931","i933","i930","i872",).shape == (4,1,1,2,2,)
	assert x2007.rank == 5
	x2022 = x999
	x2023 = x1000
	x2024 = x1001
	x2025 = x1002
	x2026 = x2024 * x2025
	assert set(x2026.index_ids) == set([])
	assert x2026.release_array().shape == ()
	assert x2026.rank == 0
	x2028 = x2023 * x2026
	assert set(x2028.index_ids) == set(["i965","i967",])
	assert x2028.release_array("i965","i967",).shape == (3,5,)
	assert x2028.rank == 2
	x2027 = x2022 * x2028
	assert set(x2027.index_ids) == set(["i966","i964","i962","i963","i965",])
	assert x2027.release_array("i966","i964","i962","i963","i965",).shape == (3,4,4,3,3,)
	assert x2027.rank == 5
	x2029 = x1005
	x2030 = x1006
	x2031 = x1007
	x2032 = x1008
	x2033 = x2031 * x2032
	assert set(x2033.index_ids) == set(["i965","i966","i973","i963","i964",])
	assert x2033.release_array("i965","i966","i973","i963","i964",).shape == (3,3,2,3,4,)
	assert x2033.rank == 5
	x2035 = x2030 * x2033
	assert set(x2035.index_ids) == set(["i977","i975","i974","i976","i965","i966","i963","i964",])
	assert x2035.release_array("i977","i975","i974","i976","i965","i966","i963","i964",).shape == (1,1,2,2,3,3,3,4,)
	assert x2035.rank == 8
	x2034 = x2029 * x2035
	assert set(x2034.index_ids) == set(["i965","i966","i963","i964",])
	assert x2034.release_array("i965","i966","i963","i964",).shape == (3,3,3,4,)
	assert x2034.rank == 4
	x2036 = x2027 * x2034
	assert set(x2036.index_ids) == set(["i962",])
	assert x2036.release_array("i962",).shape == (4,)
	assert x2036.rank == 1
	x2010 = x1013
	x2011 = x1014
	x2012 = x2010 * x2011
	assert set(x2012.index_ids) == set([])
	assert x2012.release_array().shape == ()
	assert x2012.rank == 0
	x2008 = x1015
	x2014 = x2012 * x2008
	assert set(x2014.index_ids) == set([])
	assert x2014.release_array().shape == ()
	assert x2014.rank == 0
	x2009 = x1016
	x2013 = x2014 * x2009
	assert set(x2013.index_ids) == set(["i962","i979","i931","i930","i980",])
	assert x2013.release_array("i962","i979","i931","i930","i980",).shape == (4,5,1,2,4,)
	assert x2013.rank == 5
	x2038 = x2036 * x2013
	assert set(x2038.index_ids) == set(["i979","i931","i930","i980",])
	assert x2038.release_array("i979","i931","i930","i980",).shape == (5,1,2,4,)
	assert x2038.rank == 4
	x2015 = x1019
	x2016 = x1020
	x2017 = x1021
	x2018 = x1022
	x2019 = x2017 * x2018
	assert set(x2019.index_ids) == set(["i986",])
	assert x2019.release_array("i986",).shape == (1,)
	assert x2019.rank == 1
	x2021 = x2016 * x2019
	assert set(x2021.index_ids) == set(["i988","i979","i987","i980","i932","i986",])
	assert x2021.release_array("i988","i979","i987","i980","i932","i986",).shape == (5,5,1,4,4,1,)
	assert x2021.rank == 6
	x2020 = x2015 * x2021
	assert set(x2020.index_ids) == set(["i933","i979","i980","i932",])
	assert x2020.release_array("i933","i979","i980","i932",).shape == (1,5,4,4,)
	assert x2020.rank == 4
	x2037 = x2038 * x2020
	assert set(x2037.index_ids) == set(["i931","i930","i933","i932",])
	assert x2037.release_array("i931","i930","i933","i932",).shape == (1,2,1,4,)
	assert x2037.rank == 4
	x2039 = x2007 * x2037
	assert set(x2039.index_ids) == set(["i872",])
	assert x2039.release_array("i872",).shape == (2,)
	assert x2039.rank == 1
	x2041 = x1975 * x2039
	assert set(x2041.index_ids) == set(["i754","i755","i873","i503","i756","i872",])
	assert x2041.release_array("i754","i755","i873","i503","i756","i872",).shape == (2,4,3,5,1,2,)
	assert x2041.rank == 6
	x2040 = x1944 * x2041
	assert set(x2040.index_ids) == set(["i754","i755","i503","i756",])
	assert x2040.release_array("i754","i755","i503","i756",).shape == (2,4,5,1,)
	assert x2040.rank == 4
	x2042 = x1913 * x2040
	assert set(x2042.index_ids) == set(["i3","i504","i503",])
	assert x2042.release_array("i3","i504","i503",).shape == (5,4,5,)
	assert x2042.rank == 3
	x2044 = x1786 * x2042
	assert set(x2044.index_ids) == set(["i505","i506","i4","i6","i5","i3","i504","i503",])
	assert x2044.release_array("i505","i506","i4","i6","i5","i3","i504","i503",).shape == (3,4,3,4,3,5,4,5,)
	assert x2044.rank == 8
	x2043 = x1660 * x2044
	assert set(x2043.index_ids) == set(["i4","i6","i5","i3",])
	assert x2043.release_array("i4","i6","i5","i3",).shape == (3,4,3,5,)
	assert x2043.rank == 4
	x2045 = x1532 * x2043
	assert set(x2045.index_ids) == set(["i2","i1","i0","i4","i3",])
	assert x2045.release_array("i2","i1","i0","i4","i3",).shape == (3,2,3,3,5,)
	assert x2045.rank == 5
	assert set(x0.index_ids) == set(x2045.index_ids)
	assert isclose(x0.release_normalized_array(*x0.index_ids), x2045.release_normalized_array(*x0.index_ids)).all()
	print("Test final arrays comparison: Ok")

if __name__ == "__main__":
	test_autogen()
