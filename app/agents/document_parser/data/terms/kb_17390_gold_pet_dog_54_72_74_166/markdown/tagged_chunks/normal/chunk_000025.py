from langchain_core.documents import Document

chunk = Document(
    page_content=('가지급제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자 또는\n'
 '보험수익자에게 즉시 통지합니다. 다만, 지급예정일은 다음 각 호의 어느 하나에\n'
 '해당하는 경우를 제외하고는 제7조(보험금의 청구)에서 정한 서류를 접수한 날부\n'
 '터 30영업일 이내에서 정합니다.\n'
 '1. 소송제기- \n'
 '- 2. 분쟁조정 신청\n'
 '- 3. 수사기관의 조사\n'
 '- 4. 해외에서 발생한 보험사고에 대한 조사\n'
 '- 56 -- 5. 제6항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
