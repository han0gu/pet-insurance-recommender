from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문</p><br><p id='134' "
 "data-category='paragraph' style='font-size:16px'>의 중에 정하며, 보험금 지급사유 판정에 드는 "
 "의료비용은 회사가 전액 부담합니다.</p><p id='135' data-category='list' "
 'style=\'font-size:16px\'>제3조(수술의 정의와 장소)<br>\uf000 이 특별약관에 있어서 "수술" 이라 함은 병원 '
 '또는 의원의 의사, 치과의사 면허를<br>가진 자(이하 "의사" 라 합니다)에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
