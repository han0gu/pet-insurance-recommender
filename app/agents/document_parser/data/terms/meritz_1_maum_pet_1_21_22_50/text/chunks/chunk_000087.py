from langchain_core.documents import Document

chunk = Document(
    page_content=('전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않\n'
 '은 것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우편 등) 또는 전화(음성녹\n'
 '음)로 다시 알려 드립니다.\n'
 '③ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음 각\n'
 '호의 요건을 모두 충족하는 경우에 「보험업감독규정」 제4-36조 제3항에 따른 전자\n'
 '적 상품설명장치를 활용할 수 있습니다.1. 계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등을 한'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
