from langchain_core.documents import Document

chunk = Document(
    page_content=('안내하고자 할 경우에는 계약자에게 서면 또는 전자서명법\n'
 '제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확인을\n'
 '조건으로 전자문서를 송신하여야 하며, 계약자가 전자문서\n'
 '에 대하여 수신을 확인하기 전까지는 그 전자문서는 송신되\n'
 '지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않은\n'
 '것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우\n'
 '편 등) 또는 전화(음성녹음)로 다시 알려 드립니다.\n'
 '\uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹\n'
 '음)로 안내하고자 할 때 다음 각 호의 요건을 모두 충족하'),
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
