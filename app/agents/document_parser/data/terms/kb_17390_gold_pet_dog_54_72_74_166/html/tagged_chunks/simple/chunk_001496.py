from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 각 측정치의 결과값<br>차이가 ±10dB 이상인 경우 청성뇌간반응검사(ABR)를 통해 객관적인 장<br>해 상태를 '
 '재평가하여야 한다.<br>2) ‘한 귀의 청력을 완전히 잃었을 때’라 함은 순음청력검사 결과 평균순<br>법<br>음역치가 90dB '
 '이상인 경우를 말한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001496',
              'chunk_char_len': 156,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
