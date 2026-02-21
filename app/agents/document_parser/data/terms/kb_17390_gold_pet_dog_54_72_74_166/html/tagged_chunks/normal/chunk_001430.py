from langchain_core.documents import Document

chunk = Document(
    page_content=('청약하고 회사가 승낙하여 전환대상계약이 장애인전용보험으로 전환된 경우, 이 특별약관을 청약하기 전(2022년 1월 15일~ 2022년 '
 '5월 31일)에 납입된 보험료는 해당 연도 보험료 납입영수증에 장애인전용 보장성 보험료로 표시되지 않고 특별세액공제 대상에 포함되지 '
 '않으며, 장애인전용보험으로 전환된 이후 (2022년6월1일~2022년12월31일) 납입된 보험료만 2022년 특별세액공제 '
 "대상이</td></tr></tbody></table><br><p id='99' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001430',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
