from langchain_core.documents import Document

chunk = Document(
    page_content=('- 녹음)로 다시 알려 드립니다.\n'
 '- ⑥ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음\n'
 '- 각 호의 요건을 모두 충족하는 경우에 「보험업감독규정」 제4-36조 제3항에 따른\n'
 '- 전자적 상품설명장치를 활용할 수 있습니다.\n'
 '- 1. 계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등을 한\n'
 '- 다는 사실을 미리 안내하고 동의를 받을 것\n'
 '- 2.전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두 수신하\n'
 '- 고 이해하였음을 확인할 것'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
