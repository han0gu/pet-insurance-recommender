from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 경우</p><br><p id='217' data-category='paragraph' "
 "style='font-size:16px'>회사는 변경지정을 서면으로 알리거나 보험증권에 그 뜻을 기재하여 드립니다.<br>1. "
 "청구서(회사양식)</p><br><p id='218' data-category='list' style='font-size:14px'>2. "
 '지정대리청구인의 주민등록등본<br>3'),
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
