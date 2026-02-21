from langchain_core.documents import Document

chunk = Document(
    page_content=('는 사실을 미리 안내하고 동의를 받을 것- 2.전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두 수신하\n'
 '- 고 이해하였음을 확인할 것\n'
 '- 3.계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의 활용을\n'
 '- 중단할 것을 요구하는 경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1항에 따른\n'
 '- 납입최고(독촉) 등을 실시할 것\n'
 '4.전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
