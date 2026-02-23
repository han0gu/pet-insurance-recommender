from langchain_core.documents import Document

chunk = Document(
    page_content=('- 습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하\n'
 '- 며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.\n'
 '- ⑤ 같은 상해로 두 가지 이상의 후유장해가 생긴 경우에는 후유장해 지급률을 합산하여\n'
 '- 지급합니다. 다만, 장해분류표의 각 신체부위별 판정기준에 별도로 정한 경우에는 그\n'
 '- 기준에 따릅니다.\n'
 '- ⑥ 다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경우에는 그 때마다 이에 해당\n'
 '- 하는 후유장해지급률을 결정합니다. 그러나 그 후유장해가 이미 상해 후유장해(80%'),
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
