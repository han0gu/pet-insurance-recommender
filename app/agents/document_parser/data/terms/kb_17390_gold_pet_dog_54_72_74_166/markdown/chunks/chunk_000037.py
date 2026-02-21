from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 용 어 풀 이 ∙ 적립부분 순보험료 : 적립보험료에서 사업비를 차감한 보험료 | 용 어 풀 이 ∙ 적립부분 순보험료 : 적립보험료에서 '
 '사업비를 차감한 보험료 | 관 |\n'
 '# ∙ 보험료납입일 : 보험료가 회사에 입금된 날\uf000 제4항의 공시이율은 이 보험의 사업방법서에서 정한 바에 따라 아래와 같이 '
 '결정합\n'
 '니다. 다만, 보험기간 중에 공시이율이 변경되는 경우에 변경된 시점 이후에는 변 특별\n'
 '경된 이율을 적용합니다.- 약'),
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
