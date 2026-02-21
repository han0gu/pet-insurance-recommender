from langchain_core.documents import Document

chunk = Document(
    page_content=('. 규정<br>나) 위 가)의 경우 “<붙임>일상생활 기본동작(ADLs) 제한 장해평가표”<br>상 지급률이 10% 미만인 경우에는 '
 '보장대상이 되는 장해로 인정하지<br>않는다.<br>다) 신경계의 장해로 발생하는 다른 신체부위의 장해(눈, 귀, 코, 팔,<br>다리 '
 '등)는 해당 장해로도 평가하고 그 중 높은 지급률을 적용한다.<br>라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발병 또는 외상 '
 '후 12<br>개월 동안 지속적으로 치료한 후에 장해를 평가한다'),
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
