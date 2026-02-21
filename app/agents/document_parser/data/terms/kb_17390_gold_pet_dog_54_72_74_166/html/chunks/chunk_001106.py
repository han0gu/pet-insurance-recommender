from langchain_core.documents import Document

chunk = Document(
    page_content=("KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='97' data-category='paragraph' "
 'style=\'font-size:14px\'>영업을 행하는 시설을 말합니다.<br>\uf000 제1항의 "장례서비스"라 함은 화장서비스, '
 '장례서비스, 필수 장례용품 등을 말하<br>며, 장례 이전 반려동물 사체 임시 안치, 한지·자개 등 기능성 유골함 및 수목<br>함, '
 '유골보석 제작, 봉안당(납골당) 안치는 포함하지 않습니다.<br>\uf000 제1항의 반려동물장례비용지원금은 총 장례비용에 보험증권에 '
 '기재된'),
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
