from langchain_core.documents import Document

chunk = Document(
    page_content=('. 각 상품별 사업방법서 별지는 당사 인터넷홈페이지의 상품공시실에서 확인 하실 수 있습니다. ∙ 외부지표금리 사업방법서에 정한 방법에 '
 '따라 국고채 수익률, 회사채 수익률, 통화안정증권 수익률 및 양도성예금증서 유통수익률을 기준으로 산출합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 134,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
