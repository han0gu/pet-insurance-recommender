from langchain_core.documents import Document

chunk = Document(
    page_content=('보건복지부고시<br>「장애정도판정기준」의 ‘능력장애측정기준’주 \ue045 상 6개 항목 중 3<br>개 항목 이상에서 독립적 수행이 '
 "불가능하여 타인의 도움이 필요하<br>고 GAF 50점 이하인 상태를 말한다.</p><br><p id='4' "
 "data-category='paragraph' style='font-size:16px'>주) 능력장애측정기준의 항목 : ㉮ 적절한 "
 '음식섭취, ㉯ 대소변관<br>리, 세면, 목욕, 청소 등의 청결 유지, ㉰ 적절한 대화기술 및<br>협조적인 대인관계, ㉱ 규칙적인 '
 '통원․약물 복용, ㉲ 소지품'),
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
 'indexing': {'chunk_id': 'chunk_001661',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
