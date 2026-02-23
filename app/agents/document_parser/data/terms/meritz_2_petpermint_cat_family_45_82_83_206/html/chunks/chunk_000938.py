from langchain_core.documents import Document

chunk = Document(
    page_content=(". 평형기능의 장해</h1><br><p id='25' data-category='paragraph' "
 "style='font-size:16px'>1) “평형기능에 장해를 남긴 때”라 함은 전정기관 이<br>상으로 보행 등 일상생활이 어려운 "
 "상태로 아래의 평<br>형장해 평가항목별 합산점수가 30점 이상인 경우를<br>말한다.</p><br><table id='26' "
 "style='font-size:16px'><thead><tr><td>항목</td><td>내 "
 '용</td><td>점수</td></tr></thead><tbody><tr><td'),
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
