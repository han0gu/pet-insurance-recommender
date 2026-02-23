from langchain_core.documents import Document

chunk = Document(
    page_content=('협착증 · AS</td></tr><tr><td>HAA013</td><td>폐동맥 협착 · '
 'PS</td></tr><tr><td>HAA019</td><td>기타 선천성 심장 질환 (확진된) '
 '심장사상충증</td></tr><tr><td>HAA020</td><td>기타 심혈관계 질환</td></tr><tr><td>HAA021 '
 'HAA022</td><td>점액성 이첨판막변성</td></tr><tr><td>HAA023</td><td>고양이 비대성 '
 '심근병증</td></tr><tr><td rowspan="8">4</td><td'),
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
