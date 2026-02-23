from langchain_core.documents import Document

chunk = Document(
    page_content=('- 되는 경우 납입최고(독촉)와 계약의 해지)에서 정한 계약의 해지가 발생하지 않은 경\n'
 '- 우를 말합니다.\n'
 '- ⑧ 제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에서 정한 계약의 부\n'
 '- 활이 이루어진 경우 부활을 청약한 날을 제5항의 청약일로 하여 적용합니다.\n'
 '# 제18조 (청약의 철회)① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만,\n'
 '회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융'),
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
