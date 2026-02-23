from langchain_core.documents import Document

chunk = Document(
    page_content=('- 집니다.\n'
 '| 용 어 풀 | 이 현저하게 공정을 잃은 합의 |\n'
 '| --- | --- |\n'
 '| 사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 공정 성을 잃은 것을 말합니다. | 사회통념상 일반 보통인이라면 '
 '그 같은 일을 하지 않을 정도로 현저하게 공정 성을 잃은 것을 말합니다. |\n'
 '# 제22조(재가입)# \uf000계약이 다음 각 호의 조건을 충족하고 계약자가 제4항에 따라 재가입 의사를 표시한 때에는 '
 '제11조(보험계약의 성립) 및 보통약관 제1절 일반조항 제20조(약관 교'),
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
 'indexing': {'chunk_id': 'chunk_000532',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
