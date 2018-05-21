//dat -> bin convert

//#include <stdio.h>
//#include <stdlib.h>
//#include <fstream>
//#include <sstream>
//#include <iostream>
//
//#define size 10
//using namespace std;
//
//int main()
//{
//	char file_name[] = "simplenetworkData/new/b2_new.bin";
//	float s = .9f;
//	cout.precision(11);
//	
//	std::ifstream datFile;
//	datFile.open("simplenetworkData/new/b2.bin", std::ios::out | std::ios::binary);
//	if (!datFile)
//	{
//		std::cout << "error opening file" << std::endl;
//	}
//
//	float* data_tmp = new float[size];
//	int size_b = size * sizeof(float);
//
//	datFile.read((char*)(&data_tmp), size_b);
//
//	FILE *fp;
//	int i;
//
//	if (fopen_s(&fp, file_name, "wb") != 0)
//	{
//		printf("cannot create file\n");
//		exit(1);
//	}
//	for (i = 0; i < size; i++)
//	{
//		datFile.is_open();
//		datFile >> s;
//		cout << s << endl;
//
//		fwrite(&s, sizeof(float), 1, fp);
//	}
//	fclose(fp);
//	system("pause");
//
//	return 0;
//}


//bin 파일 출력

//#include <stdio.h>
//#include <stdlib.h>
//#include <fstream>
//#include <sstream>
//#include <iostream>
//
//#define size 640
//using namespace std;
//
//int main()
//{
//	float s = .9f;
//	cout.precision(11);
//	
//	std::ifstream datFile;
//	datFile.open("simplenetworkData/fin_W2.bin", std::ios::in | std::ios::binary);
//	if (!datFile)
//	{
//		std::cout << "error opening file" << std::endl;
//	}
//
//	float data_tmp[size];
//	int size_b = size * sizeof(float);
//
//	datFile.read((char*)(data_tmp), size_b);
//
//
//	int i;
//	
//	for (i = 0; i < size; i++)
//	{
//		//datFile.is_open();
//		//datFile >> s;
//		//cout << s << endl;
//		cout << i<< " : "<< data_tmp[i] << endl;
//	}
//
//	system("pause");
//
//	return 0;
//}