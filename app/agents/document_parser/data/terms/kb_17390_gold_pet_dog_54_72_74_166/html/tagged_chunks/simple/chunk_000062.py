from langchain_core.documents import Document

chunk = Document(
    page_content=('. 통약</td></tr><tr><td colspan="2">용 어 풀 이 ∙ 적립부분 순보험료 : 적립보험료에서 사업비를 차감한 '
 "보험료</td><td>관</td></tr></tbody></table><br><h1 id='78' "
 "style='font-size:16px'>∙ 보험료납입일 : 보험료가 회사에 입금된 날</h1><br><p id='79' "
 "data-category='paragraph' style='font-size:16px'>\uf000 제4항의 공시이율은 이 보험의 "
 '사업방법서에서 정한 바에 따라 아래와 같이 결정합<br>니다'),
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
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
