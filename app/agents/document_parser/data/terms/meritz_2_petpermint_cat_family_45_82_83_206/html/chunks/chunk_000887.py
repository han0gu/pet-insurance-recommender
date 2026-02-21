from langchain_core.documents import Document

chunk = Document(
    page_content=('정체 포함)</td></tr><tr><td>KDA017</td><td>항문낭염 / 항문낭 '
 '파열</td></tr><tr><td>KDA018</td><td>항문 주위 피부염 / 항문 주위 '
 '누공</td></tr><tr><td>KEA001</td><td>식도 탈장</td></tr><tr><td>KEA003</td><td>배꼽 '
 '탈장</td></tr><tr><td>KEA004</td><td>사타구니 탈장 (서혜부 탈장 '
 '포함)</td></tr><tr><td>KEA005</td><td>회음부'),
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
