from langchain_core.documents import Document

chunk = Document(
    page_content=('혈전 / 색전증</td></tr><tr><td>기타 선천성 순환기계 '
 '질환</td></tr><tr><td>기타</td></tr><tr><td>심질환 기타 혈관 질환</td></tr><tr><td>기타 림프계 '
 '질환</td></tr><tr><td>기타 순환기계 '
 '질환</td></tr><tr><td></td></tr><tr><td>부정맥</td></tr></tbody></table><br><table '
 "id='6' style='font-size:14px'><thead><tr><td>코드</td><td>특정 질병</td><td>세부"),
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
 'indexing': {'chunk_id': 'chunk_001812',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
