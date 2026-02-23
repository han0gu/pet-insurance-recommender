from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해판정시기를 별도로 정한 경우에는 그에 따릅니다. 질\n'
 '- \uf000 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의 병\n'
 '- 효력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일부터 2년 이\n'
 '- 내로 하고, 보험기간이 10년 미만인 계약은 상해 발생일부터 1년 이내)에 장해상\n'
 '- 태가 더 악화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정합니\n'
 '- 반\n'
 '- 다. 려\n'
 '- \uf000 장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 동'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
