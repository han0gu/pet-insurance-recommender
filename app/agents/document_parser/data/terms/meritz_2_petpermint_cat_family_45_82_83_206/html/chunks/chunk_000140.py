from langchain_core.documents import Document

chunk = Document(
    page_content=("】</h1><br><p id='97' data-category='paragraph' "
 'style=\'font-size:20px\'>"전자서명"이란 다음 각 목의 사항을 나타내는 데 이용하<br>기 위하여 전자문서에 '
 "첨부되거나 논리적으로 결합된 전<br>자적 형태의 정보를 말한다.</p><br><p id='98' "
 "data-category='paragraph' style='font-size:16px'>가"),
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
