from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1조(보험금의 지급사유)에서 장해지급률이 상해 발생일부터 180일 이내에 확정\n'
 '- 되지 않는 경우에는 상해 발생일부터 180일이 되는 날의 의사 진단에 기초하여 고\n'
 '- 정될 것으로 인정되는 상태를 장해지급률로 결정합니다. 다만, 장해분류표에 장\n'
 '- 해판정시기를 별도로 정한 경우에는 그에 따릅니다.\n'
 '- \uf000 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의\n'
 '- 효력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일부터 2년 이'),
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
