from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한, 회사가 "천식지속상태"의 조사나 확인<br>을 위하여 필요하다고 인정하는 경우에는 검사결과, 진료기록부의 사본 '
 "제출을</p><br><h1 id='213' style='font-size:14px'>요청할 수 있습니다.</h1><br><table "
 "id='214' style='font-size:14px'><thead></thead><tbody><tr><td>유 의 "
 '사</td><td>항 【별표13】(천식지속상태 분류표)에서 정한 천식지속상태(J46)는 한국표준질 병 ․사인분류 질병코딩지침서(통계청)의 '
 '"주요 질환별 분류 지침"에'),
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
 'indexing': {'chunk_id': 'chunk_000663',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
