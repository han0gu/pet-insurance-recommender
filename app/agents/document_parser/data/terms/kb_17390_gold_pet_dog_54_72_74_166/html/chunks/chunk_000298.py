from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 회사는 변경 지정을 서면으로 알리거나 보험증권의 뒷면에 기재하여<br>드립니다.<br>1. 지정대리청구인 '
 "변경신청서(회사양식)</p><br><p id='132' data-category='paragraph' "
 "style='font-size:16px'>제출하고 지정대리청구인을 변경 지정할 수 있</p><p id='133' "
 "data-category='list' style='font-size:16px'>2. 지정대리청구인의 주민등록등본, 가족관계등록부 "
 '(기본증명서 등)<br>3'),
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
