from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 이 경우 청약철회는 2025년 1월 20일까지 가능합니다.\n'
 '2. 청약일 : 2025년 1월 3일 / 보험증권을 받은 날 : 2025년 1년 20일인 경우- 보험증권을 받은 날부터 15일 : '
 '2025년 2월 4일- 청약을 한 날로부터 30일 : 2025년 2월 2일 (←먼저도래)※ 이 경우 청약철회는 2025년 2월 2일까지 '
 '가능합니다.- 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표시하기 위한 서면, 전자우편, 휴대전화 문'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
