from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하는 후유장해지급률을 결정합니다. 그러나 그 후유장해가 이미 상해 후유장해(80%\n'
 '- 이상) 보험금을 지급받은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는 상\n'
 '# 경우에는 그 기준에 따릅니다.⑦ 이미 이 계약에서 상해 후유장해(80%이상) 보험금 지급사유에 해당되지 않았거나(보\n'
 '장개시 이전의 원인에 의하거나 또는 그 이전에 발생한 후유장해를 포함합니다), 상해\n'
 '후유장해(80%이상) 보험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에\n'
 '또다시 제6항에 규정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해'),
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
