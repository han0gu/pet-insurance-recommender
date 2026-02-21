from langchain_core.documents import Document

chunk = Document(
    page_content=('| KEA005 | 회음부 탈장 |  |  |\n'
 '| KEA006 | 대퇴 탈장 |  |  |\n'
 '| KEA007 | 직장탈장 |  |  |\n'
 '| KEA008 | 기타 복부탈장 |  |  |\n'
 '| KFA001 | 복막염 |  |  |\n'
 '| KGA001 | 트리코모나스증 |  |  |\n'
 '| KGA002 | 지아르디아 증 |  |  |\n'
 '| KGA003 | 콕시듐증 |  |  |\n'
 '| KGA004 | 회충증 |  |  |\n'
 '| KGA005 | 촌충증 |  |  |\n'
 '| KGA006 | 간충증 |  |  |'),
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
