from langchain_core.documents import Document

chunk = Document(
    page_content=("양쪽 고환 또는 양쪽 난소를 모두 잃었을 때</p><br><p id='22' data-category='paragraph' "
 "style='font-size:16px'>4) “흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를<br>남긴 때”라 함은 아래의 경우 중 "
 "하나에 해당하는 때<br>를 말한다.</p><br><p id='23' data-category='list' "
 "style='font-size:16px'>가) 한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때<br>나) 방광 기능상실로 영구적인 요도루, "
 '방광루, 요관<br>장문합'),
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
