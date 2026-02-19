from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 할증보험료법\n'
 '할증위험률에 의한 보험료와 표준체 보험료와의 차액을 특별약관보험료라 하며 보 험계약을 체결할 때의 위험의 정도에 따라 표준체 보험료에 '
 '회사에서 정한 특별약 관보험료(보험계약이 갱신되는 경우에는 갱신시점의 표준체 보험요율을 기준으로 이 특별약관보험료도 재산출합니다)를 '
 '더하여 납입보험료로 합니다. 다만, 특별약 관보험료는 보험계약의 보험료 납입방법에 관계없이 계약자가 보장보험료로 추가 납입하여야 합니다. '
 '이러한 경우 피보험자에게 보험사고가 발생하였을 때에는 보 험계약에 정한 보험금을 지급합니다.\n'
 '<용어풀이>'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000673',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
