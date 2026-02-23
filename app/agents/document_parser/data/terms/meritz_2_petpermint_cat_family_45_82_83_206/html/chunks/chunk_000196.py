from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 전자문서가 수신되지 않은<br>것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우<br>편 등) 또는 전화(음성녹음)로 '
 '다시 알려 드립니다.<br>\uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹<br>음)로 안내하고자 할 때 다음 각 '
 '호의 요건을 모두 충족하<br>는 경우에「보험업감독규정」제4-36조 제3항에 따른 전자적<br>상품설명장치를 활용할 수 '
 "있습니다.</p><br><p id='62' data-category='list' style='font-size:16px'>① 계약자에게 "
 '전자적 상품설명장치를'),
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
