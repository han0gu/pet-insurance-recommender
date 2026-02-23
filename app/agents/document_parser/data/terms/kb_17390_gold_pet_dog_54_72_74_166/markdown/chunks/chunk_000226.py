from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험금에서 이미 지급받은 후유장해보험금을 차감하여 지급합니다. 다만, 장해분\n'
 '72 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)류표의 각 신체부위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니다.\n'
 '\uf000 이미 이 보장에서 후유장해보험금 지급사유에 해당되지 않았거나(보장개시 이전\n'
 '의 원인에 의하거나 또는 그 이전에 발생한 후유장해를 포함합니다), 후유장해보\n'
 '험금이 지급되지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6항에 규\n'
 '정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해에 대한 후유'),
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
