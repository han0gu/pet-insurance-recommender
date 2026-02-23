from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>상</p><br><p id='93' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 발생한 급격하고도 우연한 "
 '외래<br>및<br>의 사고로 병원 또는 의원(한방병원 또는 한의원을 포함합니다)등에서 치료를 받<br>질<br>고 그 직접적인 결과로 '
 '인하여 안면부, 상지, 하지에 외형상의 반흔(흉터)이나<br>병<br>추상장해, 신체의 기형이나 기능장해가 발생하여 그 원상회복을 '
 '목적으로 사고일<br>로부터'),
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
