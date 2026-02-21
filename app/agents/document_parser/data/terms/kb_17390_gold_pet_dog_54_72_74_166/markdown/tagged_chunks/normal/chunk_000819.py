from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 이 특별약관은 다음 각 호의 어느 하나에 해당하는 경우 계약자의 청약과 회사의\n'
 '승낙으로 보험계약(보험약관을 말하며, 특별약관이 부가된 경우에는 그 특별약관\n'
 '을 포함합니다. 이하 같습니다)에 부가하여 이루어집니다. 단, 제2호에 해당하는- 경우 계약자의 동의가 필요합니다.\n'
 '- 1. 보험계약을 체결할 때 해당 반려동물의 건강상태가 보험회사가 정한 기준에 적\n'
 '- 합하지 않은 경우\n'
 '- 2. 보험계약을 체결한 후 제4장 반려동물 관련 특별약관 반려동물(강아지) 일반조\n'
 '- 항 제9조(알릴 의무 위반의 효과) 등으로 보장을 제한하는 경우'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000819',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
