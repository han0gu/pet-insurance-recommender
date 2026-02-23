from langchain_core.documents import Document

chunk = Document(
    page_content=('또다시 제6항에 규정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해\n'
 '에 대한 상해 후유장해(80%이상) 보험금이 지급된 것으로 보고 최종 후유장해 상태에\n'
 '해당되는 상해 후유장해(80%이상) 보험금에서 이를 차감하여 지급합니다.# <유의사항>동일한 부위에 다른 원인으로 후유장해가 2회이상 '
 '발생한 경우최종 장해상태에 해당하는 후유장해 보험금에서 아래 금액을 차감하여 지급합니다.- - 이전의 후유장해로 이미 지급받은 보험금이 '
 '있는 경우 그 보험금\n'
 '- - 이전의 후유장해가 보험금 지급사유에 해당되지 않은 경우라도,'),
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
