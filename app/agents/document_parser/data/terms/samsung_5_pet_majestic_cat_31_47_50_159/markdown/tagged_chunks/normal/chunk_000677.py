from langchain_core.documents import Document

chunk = Document(
    page_content=('험계약을 체결할 때의 위험의 정도에 따라 표준체 보험료에 회사에서 정한 특별약\n'
 '관보험료(보험계약이 갱신되는 경우에는 갱신시점의 표준체 보험요율을 기준으로\n'
 '이 특별약관보험료도 재산출합니다)를 더하여 납입보험료로 합니다. 다만, 특별약\n'
 '관보험료는 보험계약의 보험료 납입방법에 관계없이 계약자가 보장보험료로 추가\n'
 '납입하여야 합니다. 이러한 경우 피보험자에게 보험사고가 발생하였을 때에는 보\n'
 '험계약에 정한 보험금을 지급합니다.# <용어풀이>[할증위험률에 의한 보험료]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000677',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
