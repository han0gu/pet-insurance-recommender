from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 제1항에도 불구하고 청약한 날부터 30일(만 65세 이상의<br>계약자가 전화를 "
 '이용하여 체결한 계약은 45일로 합니다)이<br>초과된 계약은 청약을 철회할 수 없습니다.<br>\uf000 청약철회는 계약자가 전화로 '
 '신청하거나, 철회의사를 표<br>시하기 위한 서면, 전자우편, 휴대전화 문자메시지 또는 이<br>에 준하는 전자적 의사표시(이하‘서면 '
 '등’이라 합니다)를<br>발송한 때 효력이 발생합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
