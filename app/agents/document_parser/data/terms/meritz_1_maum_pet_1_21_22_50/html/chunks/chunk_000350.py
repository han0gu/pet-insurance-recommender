from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 경우<br>회사는 변경 지정을 서면으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.</p><br><p id='21' "
 "data-category='list' style='font-size:14px'>1. 지정대리청구인 변경신청서(회사양식)<br>2. "
 '지정대리청구인의 주민등록등본, 가족관계등록부(기본증명서 등)<br>3'),
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
