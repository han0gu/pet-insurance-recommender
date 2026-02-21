from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를 방지 또는 경감할 수 있\n'
 '- 었던 금액\n'
 '- 2. 제1항 제2호의 경우에는 제3자로부터 손해의 배상을 받을 수 있었던 금액\n'
 '- 3. 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호\n'
 '사비용과 회사의 동의를 받지 않은 행위에 따라 증가된 손해- 제13조(손해배상청구에 대한 회사의 해결)\n'
 '- \uf000 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 특'),
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
 'indexing': {'chunk_id': 'chunk_000685',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
