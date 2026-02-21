from langchain_core.documents import Document

chunk = Document(
    page_content=('다만, 영구히 고정된 증상은 아니지만 치료 종결 후 한시적으로 나타나는 장<br>해에 대하여는 그 기간이 5년 이상인 경우 해당 '
 '장해지급률의 20%를 장해지<br>급률로 한다.<br>5) 위 4)에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 '
 '기간(계<br>약의 효력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일<br>또는 질병의 진단확정일부터 2년 이내로 '
 '하고, 보험기간이 10년 미만인 계약<br>은 상해 발생일 또는 질병의 진단확정일부터 1년 이내)에 장해상태가 더 악<br>화된 때에는 '
 '그 악화된'),
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
