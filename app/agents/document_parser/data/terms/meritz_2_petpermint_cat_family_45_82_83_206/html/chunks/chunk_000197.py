from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>① 계약자에게 전자적 상품설명장치를 활용하여 제1항에<br>따른 납입최고(독촉) 등을 한다는 "
 '사실을 미리 안내하<br>고 동의를 받을 것<br>② 전자적 상품설명장치를 활용하여 안내한 납입최고(독<br>촉) 등을 계약자가 모두 '
 '수신하고 이해하였음을 확인<br>할 것<br>③ 계약자가 질의를 하거나 추가적인 설명을 요청하는 등<br>전자적 상품설명장치의 활용을 '
 '중단할 것을 요구하는<br>경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1<br>항에 따른 납입최고(독촉) 등을 실시할'),
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
