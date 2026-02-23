from langchain_core.documents import Document

chunk = Document(
    page_content=('자 중 2인이내에서 보험금의 대리청구인(이하, "지정대리청구인"이라 합니다)을\n'
 '지정할 수 있으며, 2인을 지정대리청구인으로 지정시 대표대리인을 지정해야 합니\n'
 '다. 또한, 지정대리청구인은 제40조(지정대리청구인의 변경지정)에 의한 변경지정\n'
 '별\n'
 '또는 보험금 청구 시에도 다음 각 호의 어느 하나에 해당하여야 합니다. 표- 1. 피보험자의 가족관계등록부상 또는 주민등록상의 배우자\n'
 '- 2. 피보험자의 3촌 이내의 친족'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000193',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
