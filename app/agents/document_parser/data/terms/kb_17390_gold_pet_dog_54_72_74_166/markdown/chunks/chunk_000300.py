from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다. 안경, 콘택트렌즈 등을 대체하기 위한 시력교정술(국민건강보험 요양급여\n'
 '- 대상 수술방법 또는 치료재료가 사용되지 않은 부분은 시력교정술로 봅니\n'
 '- 다)\n'
 '- 라. 외모개선 목적의 다리정맥류 수술\n'
 '- 4. 위생관리, 미모를 위한 성형수술(다만, 사고전 상태로의 회복을 위한 수술은\n'
 '- 보장합니다)\n'
 '- 5. 선천적 기형 및 이에 근거한 병상\n'
 '# 제4조(외모특정상해의 정의 및진단확정)\uf000 이 특별약관에 있어서 "외모특정상해"라 함은 【별표3】(외모특정상해 분류표)에\n'
 '서 정한 상해를 말합니다.'),
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
