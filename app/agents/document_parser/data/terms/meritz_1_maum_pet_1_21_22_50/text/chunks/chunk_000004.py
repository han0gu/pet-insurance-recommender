from langchain_core.documents import Document

chunk = Document(
    page_content=('사육ㆍ관리하는 고양이(猫)\n'
 '㉣ 유기동물 보호센터 등에서 사육ㆍ관리하는 개(犬) 또는 고양이(猫)사. 수의사 : 「수의사법」 제2조(정의)에 따라 수의업무를 담당하는 '
 '사람으로서 농림축\n'
 '산식품부장관의 면허를 받은 사람을 말합니다.\n'
 '아. 동물병원 : 「수의사법」 제2조(정의)에 따라 동물진료업을 하는 장소로서 「수의사\n'
 '법」 제17조에 따른 신고를 한 국내 진료기관을 말합니다.- 1 -2. 지급사유 관련 용어가. 상해: 보험기간 중에 발생한 급격하고도 '
 '우연한 외래의 사고로 반려동물이 입은 상'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
