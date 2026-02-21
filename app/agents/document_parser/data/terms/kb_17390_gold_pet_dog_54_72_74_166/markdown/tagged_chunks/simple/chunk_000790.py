from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조(보장특약의 자동갱신)의 규정에 따라 계약이 갱신되는 경우, 갱신보장특약의보장개시는 갱신일 당일로 합니다.제6조(준용규정)이 '
 '특약에서 정하지 않은 사항은 보통약관 및 해당 보장특약을 따릅니다.5. 전자서명제1조(적용대상)\n'
 '이 특별약관은 전자서명을 포함한 전자문서 작성 및 제공에 대한 사전동의(사전동의\n'
 '서를 통한 동의)를 받은 보험계약에 적용됩니다.제2조(특별약관의 체결 및효력)- \uf000 이 특별약관은 보통약관(다른 특별약관이 '
 '부가된 경우에는 그 특별약관도 포함합'),
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
 'indexing': {'chunk_id': 'chunk_000790',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
