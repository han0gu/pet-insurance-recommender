from langchain_core.documents import Document

chunk = Document(
    page_content=('- 뼈 몸통)의 만곡 정도에 따라 평가한다.\n'
 "- 가) 척추체(척추뼈 몸통)의 만곡변화는 객관적인 측정방법(Cobb's\n"
 '- Angle)에 따라 골절이 발생한 척추체(척추뼈 몸통)의 상․하 인접 정\n'
 '- 상 척추체(척추뼈 몸통)를 포함하여 측정하며, 생리적 정상만곡을\n'
 '- 고려하여 평가한다.\n'
 '- 나) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통)의 압박률, 골절의\n'
 '- 부위 등을 기준으로 판정한다. 척추체(척추뼈 몸통)의 압박률은 인\n'
 '- 접 상․하부[인접 상․하부 척추체(척추뼈 몸통)에 진구성 골절이 있거'),
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
