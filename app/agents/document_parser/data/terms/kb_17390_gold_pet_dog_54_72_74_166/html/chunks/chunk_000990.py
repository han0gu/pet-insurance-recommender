from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물의료비확장보장Ⅱ(주요치료)(강아지)</h1><p id='196' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험금의 지급사유)<br>\uf000 회사는 보험증권에 기재된 반려동물에게 이 특별약관의 "
 '보험기간 중 반려동물주요<br>치료에 대한 보장개시일(이하 반려동물주요치료보장개시일이라 합니다) 이후에 "<br>치료구분별 '
 '대상원인"(이하 사고라 합니다)이 발생하여 그 치료를 직접적인 목적<br>으로 국내에서 수의사에게 "반려동물주요치료"를 받은 경우에는 '
 '치료구분별로 각<br>각의'),
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
