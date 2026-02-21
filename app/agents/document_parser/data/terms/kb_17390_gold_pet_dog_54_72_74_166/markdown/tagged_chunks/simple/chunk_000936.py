from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상 지속적인 정신건강의학과의 치료를 받았으며, 보건복지부고시\n'
 '- 「장애정도판정기준」의 ‘능력장애측정기준’주 \ue045 상 6개 항목 중 3\n'
 '- 개 항목 이상에서 독립적 수행이 불가능하여 타인의 도움이 필요하\n'
 '- 고 GAF 50점 이하인 상태를 말한다.\n'
 '주) 능력장애측정기준의 항목 : ㉮ 적절한 음식섭취, ㉯ 대소변관\n'
 '리, 세면, 목욕, 청소 등의 청결 유지, ㉰ 적절한 대화기술 및\n'
 '협조적인 대인관계, ㉱ 규칙적인 통원․약물 복용, ㉲ 소지품 및\n'
 '금전관리나 적절한 구매행위, ㉳ 대중교통이나 일반공공시설의\n'
 '이용'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000936',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
